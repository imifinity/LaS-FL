from copy import deepcopy
import torch 
import torch.nn as nn 


def average_weights(weights):
    """ Implementation of FedAvg based on the paper:
        McMahan, B. et al. (2017) 'Communication-Efficient Learning of Deep Networks from Decentralized Data', 
        in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics. Artificial 
        Intelligence and Statistics, PMLR, pp. 1273-1282. Available at: https://proceedings.mlr.press/v54/mcmahan17a.html.
    """
    weights_avg = deepcopy(weights[0])
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
    lookahead_model = deepcopy(model)

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


class Localiser:
    def __init__(self, pretrained_state, sparsity=0.2):
        self.pretrained_state = pretrained_state
        self.sparsity = sparsity

    def compute_mask_and_deltas(self, client_state):
        mask, deltas, comm_cost = {}, {}, 0

        for name, w_pre in self.pretrained_state.items():
            if name not in client_state:
                continue
            w_client = client_state[name]
            delta = w_client - w_pre
            flat = delta.view(-1)
            k = max(1, int(self.sparsity * flat.numel()))

            if k < flat.numel():
                _, idx = torch.topk(flat.abs(), k)
                mask_flat = torch.zeros_like(flat, dtype=torch.bool)
                mask_flat[idx] = True
            else:
                mask_flat = torch.ones_like(flat, dtype=torch.bool)

            mask_tensor = mask_flat.view_as(delta)
            deltas[name] = delta * mask_tensor
            mask[name] = mask_tensor

            # count nonzero entries in mask (number of transmitted params)
            comm_cost += mask_tensor.sum().item() * 4  # float32 → 4 bytes

        return mask, deltas, comm_cost


# ---------------- Stitcher ---------------- #
class Stitcher:
    def __init__(self, pretrained_state):
        self.pretrained_state = deepcopy(pretrained_state)

    def stitch(self, client_deltas_list):
        stitched = deepcopy(self.pretrained_state)
        delta_accum = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in stitched.items()}
        counts = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in stitched.items()}

        for deltas in client_deltas_list:
            for name, d in deltas.items():
                if name not in delta_accum:
                    continue
                if not torch.is_floating_point(d):
                    continue  # skip non-float params (e.g. num_batches_tracked)
                mask = d.ne(0).to(d.device)
                delta_accum[name] += d
                counts[name] += mask.float()

        for name in stitched.keys():
            if not torch.is_floating_point(stitched[name]):
                continue  # don't try to average integer tensors

            nonzero = counts[name] > 0
            avg_delta = torch.zeros_like(delta_accum[name], dtype=torch.float32)
            avg_delta[nonzero] = delta_accum[name][nonzero] / counts[name][nonzero]
            stitched[name][nonzero] += avg_delta[nonzero]

        return stitched