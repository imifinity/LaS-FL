import argparse
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Dict, List

from data import load_datasets, FederatedSampler


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Mode args
    parser.add_argument("--data_root", type=str, default="./data/")
    parser.add_argument("--algorithm", type=str, default="fedavg")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--model_name", type=str, default="cnn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", type=int, default=5)

    # Fed args
    parser.add_argument("--non_iid", type=int, default=1)  # 0: IID, 1: Non-IID
    parser.add_argument("--dirichlet", type=float, default=0.5)  
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--participation", type=float, default=0.5)

    # Training args
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_client_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)

    # LaS args
    parser.add_argument('--graft_epochs', type=int, default=20) # was 10 from paper
    parser.add_argument('--graft_lr', type=float, default=1e-2) # was 1e-4 from paper
    parser.add_argument('--topk', type=float, default=0.2)
    parser.add_argument('--sparsity', type=float, default=0.2) # was 1e-5 from paper
    parser.add_argument('--sigmoid_bias', type=float, default=2.0)
    parser.add_argument('--l1_strength', type=float, default=1.0)

    # Extra args for baselines
    parser.add_argument("--prox_momentum", type=float, default=0.9)
    parser.add_argument("--acg_momentum", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)

    return parser.parse_args()


def average_weights(weights):
    """ Implementation of FedAvg based on the paper:
        McMahan, B. et al. (2017) 'Communication-Efficient Learning of Deep Networks from Decentralized Data', 
        in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics. Artificial 
        Intelligence and Statistics, PMLR, pp. 1273-1282. Available at: https://proceedings.mlr.press/v54/mcmahan17a.html.
    """
    weights_avg = copy.deepcopy(weights[0])
    total_bytes = 0

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))

        # Count bytes for this parameter across all clients
        total_bytes += weights_avg[key].numel() * weights_avg[key].element_size() * len(weights)

    return weights_avg, total_bytes

""" 
Implementation of FedAcg based on the paper:
Kim, G., Kim, J. and Han, B. (2024) 'Communication-Efficient Federated Learning with Accelerated 
Client Gradient', in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 
2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Seattle, WA, USA: 
IEEE, pp. 12385-12394. Available at: https://doi.org/10.1109/CVPR52733.2024.01177.
"""

def FedACG_lookahead(model: nn.Module, prev_global_state: dict, acg_momentum: float) -> nn.Module:
    """ 
    Compute lookahead model

    Args:
        model (nn.Module): 
        prev_global_state (dict):
        acg_momentum (float):

    Returns:
        lookahead_model (nn.Module): The new lookahead model sent to clients.
    """
    lookahead_model = copy.deepcopy(model)

    for (name, param) in lookahead_model.named_parameters():
        prev_param = prev_global_state[name]
        param.data += acg_momentum * (param.data - prev_param.data)

    return lookahead_model


def FedACG_aggregate(sending_model: nn.Module, clients_models: list[dict]) -> tuple[dict, int]:
    """
    Aggregate client updates for FedACG.

    Args:
        sending_model (nn.Module): The lookahead model sent to clients.
        clients_models (list[dict]): List of client state_dicts after local training.

    Returns:
        updated_weights (dict): The new global model weights.
        total_bytes (int): Total communication in bytes (server->clients + clients->server).
    """
    sending_state = sending_model.state_dict()

    # Compute deltas relative to the lookahead model
    clients_deltas = [
        {k: client_state[k] - sending_state[k] for k in sending_state}
        for client_state in clients_models
    ]

    # Aggregate deltas (element-wise mean)
    aggregated_delta = {}
    for k in sending_state: 
        if clients_deltas[0][k].dtype in (torch.float32, torch.float64):
            aggregated_delta[k] = torch.mean(torch.stack([delta[k] for delta in clients_deltas]), dim=0)
        else:
            aggregated_delta[k] = clients_deltas[0][k]  # just take first client's value

    # Compute updated global weights
    updated_weights = {k: sending_state[k] + aggregated_delta[k] for k in sending_state}

    # Compute communication bytes
    lookahead_bytes = sum(param.numel() * param.element_size() for param in sending_model.parameters())
    deltas_bytes = sum(sum(d.numel() * d.element_size() for d in delta.values()) for delta in clients_deltas)
    total_bytes = lookahead_bytes + deltas_bytes

    return updated_weights, total_bytes


def FedDyn_aggregate(local_weights, global_weights, histories, selected_clients, alpha):
    """
    FedDyn aggregation function based on Acar et al., 2021.
    """

    # Ensure global weights are float and record their device
    global_weights_float = {k: v.float() for k, v in global_weights.items()}
    gw_device = next(iter(global_weights_float.values())).device

    # Average local client models (per-key)
    avg_weights = {k: v.float().clone().to(gw_device) for k, v in local_weights[0].items()}
    for key in avg_weights.keys():
        for i in range(1, len(local_weights)):
            avg_weights[key] += local_weights[i][key].float().to(gw_device)
        avg_weights[key] /= float(len(local_weights))

    # Update client histories: h_i <- h_i + (w_i - w)
    for i, client_idx in enumerate(selected_clients):
        # Create history if missing (float, on correct device)
        if histories[client_idx] is None:
            histories[client_idx] = {k: torch.zeros(v.size(), dtype=torch.float32, device=gw_device)
                                     for k, v in global_weights_float.items()}
        else:
            # Ensure dtype and device match global_weights_float
            for key in global_weights_float.keys():
                h_tensor = histories[client_idx][key]
                if h_tensor.dtype != torch.float32 or h_tensor.device != gw_device:
                    histories[client_idx][key] = h_tensor.to(device=gw_device).float()

        # Now safe to add
        for key in global_weights_float.keys():
            histories[client_idx][key] += (local_weights[i][key].float().to(gw_device) - global_weights_float[key])

    # Compute mean of histories (per-key)
    hist_mean = {k: torch.zeros_like(v, dtype=torch.float32, device=gw_device) for k, v in global_weights_float.items()}
    num_hist = 0
    for h in histories:
        if h is not None:
            for key in h.keys():
                hist_mean[key] += h[key]
            num_hist += 1
    if num_hist > 0:
        for key in hist_mean.keys():
            hist_mean[key] /= float(num_hist)

    # Compute new global weights
    new_global_weights = {k: (avg_weights[k] + hist_mean[k]).to(gw_device) for k in avg_weights.keys()}

    # Estimate communication cost
    total_bytes = 0
    for key in new_global_weights.keys():
        total_bytes += new_global_weights[key].numel() * new_global_weights[key].element_size() * len(local_weights)

    return new_global_weights, histories, total_bytes



# class Localiser(nn.Module):
#     """ Implementation of Localise-and-Stitch based on the paper:
#         He, Y. et al. (2024) 'Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic'. arXiv. 
#         Available at: https://doi.org/10.48550/arXiv.2408.13656.
#     """

#     def __init__(self, trainable_params, model, pretrained_model, finetuned_model, graft_args):
#         super(Localiser, self).__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         self.params = {k: p.to(self.device) for k, p in trainable_params.items()}
#         self.model = copy.deepcopy(model).to(self.device).eval()
#         self.pretrained = copy.deepcopy(pretrained_model).to(self.device).eval()
#         self.finetuned = copy.deepcopy(finetuned_model).to(self.device).eval()
#         self.graft_args = graft_args
        
#         for p in self.pretrained.parameters():
#             p.requires_grad = False
#         for p in self.finetuned.parameters():
#             p.requires_grad = False

#         self.trainable_name = list(self.params.keys())
#         self.task_vectors = [(dict(self.finetuned.named_parameters())[n] -
#                               dict(self.pretrained.named_parameters())[n]).detach()
#                              for n in self.trainable_name]

#         self.num_params = sum(p.numel() for p in self.task_vectors)
#         self.create_basepatch()

#     def reset_model(self):
#         with torch.no_grad():
#             for name in self.trainable_name:
#                 pretensor = dict(self.pretrained.named_parameters())[name].to(self.device)
#                 self.params[name].data.copy_(pretensor)  

#     def create_basepatch(self):
#         # top-k sparsity mask
#         abs_tv = torch.cat([torch.abs(tv).view(-1) for tv in self.task_vectors])
#         k = int(self.graft_args.sparsity * abs_tv.numel())
#         topk_values, _ = torch.topk(abs_tv, k)
#         threshold = topk_values[-1]

#         self.mask = []
#         for tv in self.task_vectors:
#             mask = torch.where(torch.abs(tv) > threshold,
#                                torch.full_like(tv, self.graft_args.sigmoid_bias),
#                                torch.full_like(tv, -self.graft_args.sigmoid_bias))
#             self.mask.append(mask.to(self.device))

#     def interpolate_model(self, round_=False, return_mask=False):  
#         sigmoid = torch.nn.Sigmoid()
#         binary_mask = []
#         total_bytes = 0

#         self.reset_model()

#         with torch.no_grad():
#             for i, name in enumerate(self.trainable_name):
#                 pretensor = dict(self.pretrained.named_parameters())[name].to(self.device)
#                 finetensor = dict(self.finetuned.named_parameters())[name].to(self.device)

#                 frac = sigmoid(self.mask[i])
#                 if round_:
#                     frac = torch.round(frac)
#                     binary_mask.append(frac)

#                 delta = (finetensor - pretensor) * frac
#                 self.params[name].data.copy_(pretensor + delta)
#                 self.model.state_dict()[name].copy_(pretensor + delta)

#                 active = frac.nonzero().size(0)
#                 total_bytes += active * delta.element_size()
#                 total_bytes += active * frac.element_size()

#         if return_mask:
#             return binary_mask, sum(frac.sum() for frac in self.mask) / self.num_params, total_bytes
#         else:
#             return sum(frac.sum() for frac in self.mask) / self.num_params, total_bytes

#     def evaluate(self, dataloader):
#         self.model.eval()
#         correct, total = 0, 0

#         with torch.no_grad():
#             for x, y in dataloader:
#                 x = x.to(self.device)
#                 y = y.to(self.device)
#                 outputs = self.model(x)
#                 preds = outputs.argmax(dim=1)
#                 correct += (preds == y).sum().item()
#                 total += y.size(0)

#         return correct / total

#     def train_graft(self, dataloader):
#         loss_fct = nn.CrossEntropyLoss()
#         sigmoid = torch.nn.Sigmoid()
#         lr = self.graft_args.graft_lr

#         print("Grafting", end="")

#         for epoch in range(self.graft_args.graft_epochs):
#             print("|", end="")
#             total_grad = None
#             self.interpolate_model(round_=False)  # prepares model for grafting

#             for x, y in dataloader:
#                 x = x.to(self.device)
#                 y = y.to(self.device)
#                 outputs = self.model(x)
#                 loss = loss_fct(outputs, y)
#                 loss.backward()
                
#                 grad = [p.grad.detach().clone() for n, p in self.model.named_parameters()
#                         if n in self.trainable_name]
#                 grad = [g * tv.to(self.device) for g, tv in zip(grad, self.task_vectors)]

#                 if total_grad is None:
#                     total_grad = [lr * g for g in grad]
#                 else:
#                     total_grad = [tg + lr * g for tg, g in zip(total_grad, grad)]

#                 self.model.zero_grad()

#             total_grad = [g / len(dataloader) for g in total_grad]
#             self.reset_model()

#             with torch.no_grad():
#                 for i in range(len(self.mask)):
#                     p = self.mask[i]
#                     g = total_grad[i]
#                     deriv = sigmoid(p) * (1 - sigmoid(p))
#                     reg_term = self.graft_args.l1_strength * torch.where(p > 0, deriv, -deriv)
#                     p -= g * deriv - reg_term

#             # Evaluation of current mask
#             if (epoch + 1) % 5 == 0 or epoch == self.graft_args.graft_epochs - 1:
#                 mask, proportion, total_bytes = self.interpolate_model(round_=True, return_mask=True)
#                 acc = self.evaluate(dataloader)
#                 self.reset_model()

#         final_mask, proportion, total_bytes = self.interpolate_model(round_=True, return_mask=True)
#         self.reset_model()

#         print("\n Finished grafting")

#         return final_mask, proportion, acc, total_bytes


class Localiser(nn.Module):
    """Localiser compatible with FL.

    Updates mask *logits* in-place using the correct chain-rule for gradients so that
    interpolate_model(...) (which uses sigmoid(mask) as fractional masks) produces
    meaningful interpolated weights that Stitcher can aggregate.
    """

    def __init__(self, trainable_params, model, pretrained_model, finetuned_model, graft_args):
        super(Localiser, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # store references on device
        self.params = {k: p.to(self.device) for k, p in trainable_params.items()}
        self.model = copy.deepcopy(model).to(self.device).eval()
        self.pretrained = copy.deepcopy(pretrained_model).to(self.device).eval()
        self.finetuned = copy.deepcopy(finetuned_model).to(self.device).eval()
        self.graft_args = graft_args

        # ensure pretrained/finetuned are not trainable
        for p in self.pretrained.parameters():
            p.requires_grad = False
        for p in self.finetuned.parameters():
            p.requires_grad = False

        # names & task vectors (finetuned - pretrained)
        self.trainable_name = list(self.params.keys())
        self.task_vectors = [
            (dict(self.finetuned.named_parameters())[n].to(self.device) -
             dict(self.pretrained.named_parameters())[n].to(self.device)).detach()
            for n in self.trainable_name
        ]

        self.num_params = sum(tv.numel() for tv in self.task_vectors)

        # mask logits (not sigmoid outputs). Initialize via top-k basepatch
        self._create_basepatch()

    def _create_basepatch(self):
        # top-k by magnitude of task vector → create initial logits (positive bias where selected)
        abs_tv = torch.cat([tv.abs().view(-1) for tv in self.task_vectors])
        k = int(self.graft_args.sparsity * abs_tv.numel())
        if k < 1:
            k = 1
        topk_values, _ = torch.topk(abs_tv, k)
        threshold = topk_values[-1]

        self.mask = []
        for tv in self.task_vectors:
            # logits set to +bias where |tv| > threshold, else -bias
            m = torch.where(tv.abs() > threshold,
                            torch.full_like(tv, self.graft_args.sigmoid_bias, device=self.device),
                            torch.full_like(tv, -self.graft_args.sigmoid_bias, device=self.device))
            # keep as float tensor on device; these are the **logits** we will update
            self.mask.append(m.clone().to(self.device))

    def reset_model(self):
        # copy pretrained weights into params and model (so interpolate_model starts from pretrained)
        with torch.no_grad():
            for name in self.trainable_name:
                pretensor = dict(self.pretrained.named_parameters())[name].to(self.device)
                self.params[name].data.copy_(pretensor)
                # also ensure model's state_dict agrees
                self.model.state_dict()[name].copy_(pretensor)

    def interpolate_model(self, round_=False, return_mask=False):
        """Apply sigmoid(mask_logit) fractional masks to produce interpolated weights.

        If round_ True, also produce a binary mask list (rounded sigmoid).
        Returns: (binary_mask_list, proportion, total_bytes) when return_mask True,
                 else (proportion, total_bytes)
        """
        sigmoid = torch.nn.Sigmoid()
        binary_mask = []
        total_bytes = 0

        self.reset_model()

        with torch.no_grad():
            for i, name in enumerate(self.trainable_name):
                pretensor = dict(self.pretrained.named_parameters())[name].to(self.device)
                finetensor = dict(self.finetuned.named_parameters())[name].to(self.device)
                s = sigmoid(self.mask[i])           # fractional mask in (0,1)
                if round_:
                    s_rounded = torch.round(s)
                    binary_mask.append(s_rounded)

                delta = (finetensor - pretensor) * s
                # write both into the params mapping and the working model
                self.params[name].data.copy_(pretensor + delta)
                self.model.state_dict()[name].copy_(pretensor + delta)

                # bytes: count entries where fractional mask is non-zero (conservative)
                active = int((s.abs() > 0.0).sum().item())
                total_bytes += active * delta.element_size()
                total_bytes += active * s.element_size()

        if return_mask:
            # proportion: total grafted params over all trainable params (rounded mask)
            if len(binary_mask) > 0:
                total_ones = sum(b.sum().item() for b in binary_mask)
                prop = float(total_ones) / float(self.num_params)
            else:
                prop = float(sum(torch.round(torch.sigmoid(m)).sum().item() for m in self.mask)) / float(self.num_params)
            return binary_mask, prop, total_bytes
        else:
            total_frac = float(sum(torch.sigmoid(m).sum().item() for m in self.mask))
            return total_frac / float(self.num_params), total_bytes

    def evaluate(self, dataloader):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    def train_graft(self, dataloader):
        """Train mask logits via gradient signal coming from the model loss.

        Uses chain-rule: dL/dp = dL/dw * (s*(1-s) * task_vector), where p is mask logit and s = sigmoid(p).
        Aggregates gradients over batches, averages, and updates mask logits in-place.
        """

        loss_fct = nn.CrossEntropyLoss()
        sigmoid = torch.nn.Sigmoid()
        lr = float(self.graft_args.graft_lr)
        l1_strength = float(self.graft_args.l1_strength)
        n_batches = max(1, len(dataloader))

        print("Grafting", end="")

        last_acc = 0.0

        for epoch in range(self.graft_args.graft_epochs):
            print("|", end="", flush=True)
            # Prepare model weights according to current mask
            self.interpolate_model(round_=False)
            # Zero any grads
            self.model.zero_grad()

            # accumulators for gradient of mask logits (same shapes as masks)
            total_dlogits = [torch.zeros_like(m, device=self.device) for m in self.mask]

            # Loop over batches and accumulate dL/dp
            for xb, yb in dataloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                # forward on the interpolated model
                outputs = self.model(xb)
                loss = loss_fct(outputs, yb)
                loss.backward()  # fills grads in self.model parameters (dL/dw)

                # collect dL/dw for trainable params and compute dL/dp using chain rule
                for i, name in enumerate(self.trainable_name):
                    # gradient wrt weight w: same shape as param
                    param = dict(self.model.named_parameters())[name]
                    if param.grad is None:
                        # no gradient for this param in this batch
                        grad_w = torch.zeros_like(param.data, device=self.device)
                    else:
                        grad_w = param.grad.detach().clone().to(self.device)

                    # s*(1-s) factor
                    s = sigmoid(self.mask[i])
                    deriv_s = s * (1.0 - s)  # same shape as mask

                    # task_vector (finetuned - pretrained)
                    tv = self.task_vectors[i]

                    # dL/dp elementwise: grad_w * (deriv_s * tv)
                    dlogit = grad_w * (deriv_s * tv)

                    # accumulate
                    total_dlogits[i] += dlogit

                # clear model grads for next batch
                self.model.zero_grad()

            # Average over batches
            total_dlogits = [g / float(n_batches) for g in total_dlogits]

            # Update mask logits with gradient step and L1-style regularisation on mask magnitude
            with torch.no_grad():
                for i in range(len(self.mask)):
                    # gradient descent step (we move against dL/dp)
                    step = lr * total_dlogits[i]
                    # L1-like regularizer on the sigmoid output (encourage sparsity)
                    s = sigmoid(self.mask[i])
                    # derivative of L1(s) w.r.t p ~ sign(s) * s*(1-s) ; approximate with +/- deriv
                    reg_term = l1_strength * torch.where(self.mask[i] > 0, deriv_s, -deriv_s)
                    # update: p <- p - step - reg_term
                    self.mask[i] -= step + reg_term

            # Optionally evaluate intermediate masks
            if (epoch + 1) % 5 == 0 or epoch == self.graft_args.graft_epochs - 1:
                binary_mask, prop, total_bytes = self.interpolate_model(round_=True, return_mask=True)
                last_acc = self.evaluate(dataloader)
                # reset model weights after evaluation
                self.reset_model()
                print(f" [epoch {epoch+1}] graft_prop={prop:.6f} acc={last_acc:.4f}", end="", flush=True)

        # Final binary mask, proportion and bytes
        final_mask, proportion, total_bytes = self.interpolate_model(round_=True, return_mask=True)
        # reset model to pretrained state (so further training of client isn't contaminated)
        self.reset_model()

        print("\n Finished grafting")
        return final_mask, proportion, last_acc, total_bytes



class Stitcher(nn.Module):
    """ 
        Implementation of Localize-and-Stitch based on the paper:
        He, Y. et al. (2024) 'Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic'. arXiv. 
        Available at: https://doi.org/10.48550/arXiv.2408.13656.
    """

    def __init__(self, trainable_params, pretrained_model, finetuned_models, masks):
        super(Stitcher, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.params = {k: p.to(self.device) for k, p in trainable_params.items()}
        self.pretrained = copy.deepcopy(pretrained_model).to(self.device)
        self.finetuned_models = [copy.deepcopy(fm).to(self.device) for fm in finetuned_models]

        # Simplified average masks
        num_layers = len(masks[0])
        self.masks = []
        for k in range(num_layers):
            avg_mask = sum(mask[k] for mask in masks) / len(masks)
            self.masks.append(avg_mask.to(self.device))

        self.model = copy.deepcopy(pretrained_model).to(self.device)

#     def interpolate_models(self):
#         """Combine models using average reciprocal masks."""
#         trainable_names = list(self.params.keys())
#         total_bytes = 0

#         with torch.no_grad():
#             for fm in self.finetuned_models:
#                 for idx, name in enumerate(trainable_names):
#                     pretensor = dict(self.pretrained.named_parameters())[name]
#                     finetensor = dict(fm.named_parameters())[name]
#                     mask = (self.masks[idx] > 0.5).float()
#                     delta = (finetensor - pretensor) * mask

#                     # Update the stitched model weights directly
#                     self.model.state_dict()[name].copy_(pretensor + delta)
#                     self.params[name].data.copy_(pretensor + delta)

#                     # Estimate active bytes for delta and mask
#                     active = mask.nonzero().size(0)
#                     total_bytes += active * delta.element_size()
#                     total_bytes += active * mask.element_size()

#         return self.model, total_bytes  


    def interpolate_models(self):
        """
        Combine models by averaging masked deltas across clients and updating BN buffers.
        Replaces previous behavior that overwrote weights per client.
        """
        trainable_names = list(self.params.keys())
        total_bytes = 0
        device = self.device

        # Precompute pretrained params and masks on device
        pretensors = {name: dict(self.pretrained.named_parameters())[name].to(device) for name in trainable_names}
        masks = [m.to(device) for m in self.masks]  # self.masks is list of layer masks (one per layer)

        # Handle buffers (e.g., running_mean, running_var) by averaging across clients
        buffer_names = [n for n, _ in self.pretrained.named_buffers()]
        if buffer_names:
            # collect buffers per client and average
            avg_buffers = {}
            for buf_name in buffer_names:
                accum = None
                for fm in self.finetuned_models:
                    fm_buf = dict(fm.named_buffers()).get(buf_name, None)
                    if fm_buf is None:
                        raise RuntimeError(f"Buffer {buf_name} not found in a finetuned model.")
                    fm_buf = fm_buf.to(device)
                    if accum is None:
                        accum = fm_buf.clone().detach().to(device)
                    else:
                        accum = accum + fm_buf
                avg_buffers[buf_name] = (accum / len(self.finetuned_models)).to(device)

        with torch.no_grad():
            # Update trainable parameters by aggregating deltas across clients
            for idx, name in enumerate(trainable_names):
                pretensor = pretensors[name]
                # create accumulator for delta with same dtype/device
                agg_delta = torch.zeros_like(pretensor, device=device)

                # Debug: show pretensor stats
                print(f"[DEBUG] Layer {idx} - {name} PRE mean={pretensor.mean().item():.6f}, std={pretensor.std().item():.6f}")

                # Sum masked deltas from each finetuned model
                for fm_idx, fm in enumerate(self.finetuned_models):
                    finetensor = dict(fm.named_parameters())[name].to(device)
                    # use fractional averaged mask (not thresholded) stored in self.masks per-layer
                    mask_frac = masks[idx]  # already averaged in __init__
                    # ensure mask shape matches
                    if mask_frac.shape != pretensor.shape:
                        # if somehow mask stored transposed/flattened, raise informative error
                        raise RuntimeError(f"Mask shape mismatch for layer {name}: mask {mask_frac.shape}, param {pretensor.shape}")
                    delta = (finetensor - pretensor) * mask_frac
                    agg_delta += delta

                    # count active entries for bytes accounting (consider non-zero entries of thresholded mask)
                    active = int((mask_frac.abs() > 0.0).sum().item())
                    total_bytes += active * delta.element_size()
                    total_bytes += active * mask_frac.element_size()

                    # Debug: per-client layer stats
                    print(f"[DEBUG]   client {fm_idx} | finet mean={finetensor.mean().item():.6f}, delta mean={delta.mean().item():.6f}, mask_nonzero={active}")

                # Average across clients
                agg_delta = agg_delta / len(self.finetuned_models)

                # Construct stitched param and write to model & params dict
                stitched = pretensor + agg_delta
                # Debug: stitched stats
                print(f"[DEBUG]   AGG stitched mean={stitched.mean().item():.6f}, std={stitched.std().item():.6f}, max={stitched.max().item():.6f}, min={stitched.min().item():.6f}")

                # Write to model and exported params:
                # Use state_dict copy_ to ensure exact same key update
                self.model.state_dict()[name].copy_(stitched)
                # Also update the params mapping that fedalg uses
                self.params[name].data.copy_(stitched)

            # Update averaged buffers into stitched model (so BN running_mean/var are sensible)
            if buffer_names:
                print("[DEBUG] Updating buffers (running_mean/var) by client-average")
                for buf_name, avg_buf in avg_buffers.items():
                    # Copy averaged buffer into stitched model's state_dict (buffers are present in state_dict)
                    if buf_name in self.model.state_dict():
                        self.model.state_dict()[buf_name].copy_(avg_buf)
                        # Debug
                        b = self.model.state_dict()[buf_name]
                        print(f"[DEBUG]   Buffer {buf_name} | mean={b.mean().item():.6f}, std={b.std().item():.6f}")
                    else:
                        # Some models may name buffers differently; warn but continue
                        print(f"[DEBUG]   WARNING: buffer {buf_name} not present in stitched model.state_dict()")

        return self.model, total_bytes