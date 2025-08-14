from typing import Any, Dict, List
import argparse
import os
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import load_datasets, FederatedSampler

def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./data/")
    parser.add_argument("--algorithm", type=str, default="fedavg")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--model_name", type=str, default="cnn")

    parser.add_argument("--non_iid", type=int, default=1)  # 0: IID, 1: Non-IID
    parser.add_argument("--dirichlet", type=float, default=0.5)  
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--frac", type=float, default=0.1)

    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--n_client_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=1)

    parser.add_argument("--device", type=int, default=0)

    parser.add_argument('--graft_epochs', type=int, default=5)
    parser.add_argument('--graft_lr', type=float, default=0.01)
    parser.add_argument('--topk', type=float, default=0.2)
    parser.add_argument('--sparsity', type=float, default=0.2)
    parser.add_argument('--sigmoid_bias', type=float, default=2.0)
    parser.add_argument('--l1_strength', type=float, default=0.0)

    return parser.parse_args()


def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg


class Localiser(nn.Module):
    """ Implementation of FedSoup based on the paper:
        He, Y. et al. (2024) 'Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic'. arXiv. 
        Available at: https://doi.org/10.48550/arXiv.2408.13656.
    """

    def __init__(self, trainable_params, model, pretrained_model, finetuned_model, graft_args):
        super(Localiser, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.params = trainable_params
        self.model = model
        self.pretrained = pretrained_model
        self.finetuned = finetuned_model
        self.graft_args = graft_args

        self.model.to(self.device).eval()
        self.pretrained.to(self.device).eval()
        self.finetuned.to(self.device).eval()
        
        for p in self.pretrained.parameters():
            p.requires_grad = False
        for p in self.finetuned.parameters():
            p.requires_grad = False

        self.create_binary_masks()
        self.mask = self.create_basepatch()

    def create_binary_masks(self):
        
        self.trainable_name = list(self.params.keys())
        self.trainable_parameters = [torch.rand_like(p.data, device=self.device, requires_grad=False)
                                     for p in self.params.values()]
        self.num_params = sum(p.numel() for p in self.trainable_parameters)

        self.task_vectors = []
        for name in self.trainable_name:
            pretensor = dict(self.pretrained.named_parameters())[name]
            finetensor = dict(self.finetuned.named_parameters())[name]
            self.task_vectors.append((finetensor - pretensor).detach())

    def reset_model(self):
        with torch.no_grad():
            for name in self.trainable_name:
                pretensor = dict(self.pretrained.named_parameters())[name].to(self.device)
                self.params[name].data.copy_(pretensor)  

    def create_basepatch(self):
        abs_tv = torch.cat([torch.abs(tv).view(-1) for tv in self.task_vectors])
        k = int(self.graft_args.sparsity * abs_tv.numel())
        topk_values, _ = torch.topk(abs_tv, k)
        threshold = topk_values[-1]

        basepatch = []
        for tv in self.task_vectors:
            mask = torch.where(torch.abs(tv) > threshold,
                               torch.full_like(tv, self.graft_args.sigmoid_bias),
                               torch.full_like(tv, -self.graft_args.sigmoid_bias))
            basepatch.append(mask)

        result = sum(torch.sum(torch.round(torch.sigmoid(p))) for p in basepatch) / self.num_params
        print(f"Total parameters in mask: {result:.6f}")
              
        return basepatch

    def interpolate_model(self, round_=False, return_mask=False):  
        sigmoid = torch.nn.Sigmoid()
        binary_mask = []
        n_graft_params = 0

        with torch.no_grad():
            for i, name in enumerate(self.trainable_name):
                pretensor = dict(self.pretrained.named_parameters())[name].to(self.device)
                finetensor = dict(self.finetuned.named_parameters())[name].to(self.device)

                frac = sigmoid(self.mask[i])
                if round_:
                    frac = torch.round(frac)
                    binary_mask.append(frac)

                n_graft_params += frac.sum()
                self.params[name].data.add_(frac * (finetensor - pretensor))

        if return_mask:
            return binary_mask, n_graft_params / self.num_params

    def evaluate(self, dataloader):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in tqdm(dataloader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Mask evaluation Accuracy: {acc*100:.2f}%")

    def train_graft(self, dataloader):
        loss_fct = nn.CrossEntropyLoss()
        sigmoid = torch.nn.Sigmoid()
        lr = self.graft_args.graft_lr

        for epoch in range(self.graft_args.graft_epochs):
            print(f"Graft epoch {epoch+1}/{self.graft_args.graft_epochs}")
            total_grad = None
            self.interpolate_model(round_=False)

            for batch in dataloader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x)
                loss = loss_fct(outputs, y)
                loss.backward()
                
                grad = [p.grad.detach().clone() for n, p in self.model.named_parameters()
                        if n in self.trainable_name]
                assert len(grad) == len(self.task_vectors), "Gradient and task vector length mismatch"
                grad = [g * tv.to(self.device) for g, tv in zip(grad, self.task_vectors)]

                if total_grad is None:
                    total_grad = [lr * g for g in grad]
                else:
                    total_grad = [tg + lr * g for tg, g in zip(total_grad, grad)]

                self.model.zero_grad()

            print("2Task vector norms:", [tv.norm().item() for tv in self.task_vectors])

            total_grad = [g / len(dataloader) for g in total_grad]
            self.reset_model()

            with torch.no_grad():
                for i in range(len(self.mask)):
                    p = self.mask[i]
                    g = total_grad[i]
                    deriv = sigmoid(p) * (1 - sigmoid(p))
                    reg_term = self.graft_args.l1_strength * torch.where(p > 0, deriv, -deriv)
                    p -= g * deriv - reg_term

            # Evaluation of current mask
            if (epoch + 1) % 5 == 0 or epoch == self.graft_args.graft_epochs - 1:
                mask, proportion = self.interpolate_model(round_=True, return_mask=True)
                acc = self.evaluate(dataloader)
                self.reset_model()

        final_mask, proportion = self.interpolate_model(round_=True, return_mask=True)
        self.reset_model()

        print("Mask parameter norms:", [p.norm().item() for p in self.mask])

        return final_mask, proportion, acc


class Stitcher(nn.Module):
    """ 
        Implementation of Localize-and-Stitch based on the paper:
        He, Y. et al. (2024) 'Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic'. arXiv. 
        Available at: https://doi.org/10.48550/arXiv.2408.13656.
    """

    def __init__(self, trainable_params, pretrained_model, finetuned_models, masks):
        super(Stitcher, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.params = trainable_params
        self.pretrained = pretrained_model
        self.finetuned_models = finetuned_models
        self.masks = self.get_average_masks(masks) 

        self.model = self._clone_model(pretrained_model) # Stores interpolated weights

    def _clone_model(self, model):
        """Clone a model structure and copy initial weights."""
        return copy.deepcopy(model).to(self.device)

    def get_average_masks(self, masks):
        """Compute averaged reciprocal masks based on client overlaps."""
        def reciprocal_with_zero(tensor):
            mask = tensor == 0
            reciprocal = torch.reciprocal(tensor)
            reciprocal = reciprocal.masked_fill(mask, 0)
            return reciprocal

        output_masks = []
        num_clients = len(masks)
        num_layers = len(masks[0])

        for i in range(num_clients):
            combined = [masks[i][k].clone() for k in range(num_layers)]
            for j in range(num_clients):
                if i == j:
                    continue
                for k in range(num_layers):
                    combined[k] += torch.logical_and(masks[i][k], masks[j][k])
            for k in range(num_layers):
                combined[k] = reciprocal_with_zero(combined[k])
            output_masks.append(combined)

        return output_masks

    def interpolate_models(self):
        """Combine models using average reciprocal masks."""
        trainable_names = list(self.params.keys())
        self.model.to(self.device)
        self.pretrained.to(self.device)

        for fm, client_mask in zip(self.finetuned_models, self.masks):
            fm.to(self.device)

            for idx, name in enumerate(trainable_names):
                pretensor = dict(self.pretrained.named_parameters())[name].to(self.device)
                finetensor = dict(fm.named_parameters())[name].to(self.device)

                with torch.no_grad():
                    delta = (finetensor - pretensor) * client_mask[idx].to(self.device)
                    self.params[name].data.add_(delta)

        return self.model


def average_fedsoup_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """ 
    Implementation of FedSoup based on the paper:
    Chen, M. et al. (2023) 'FedSoup: Improving Generalization and Personalization in Federated Learning via Selective Model Interpolation'. arXiv. 
    Available at: https://doi.org/10.48550/arXiv.2307.10507.
    """
    pass