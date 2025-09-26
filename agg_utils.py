from copy import deepcopy
import torch 
import torch.nn as nn 


"""
Implementation of Fedvg aggregation. Adapted from the authors' reference code:
https://github.com/naderAsadi/FedAvg

Original algorithm introduced in:
McMahan, B. et al. (2017) 'Communication-Efficient Learning of Deep Networks from Decentralized Data', 
in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics. 
Artificial Intelligence and Statistics, PMLR, pp. 1273-1282. 
Available at: https://proceedings.mlr.press/v54/mcmahan17a.html
"""

def average_weights(weights):
    """
    Compute aggregated weights using simple averaging.

    Args:
        weights (list[dict]): List of model state_dicts from clients.

    Returns:
        tuple:
            - dict: Averaged global model weights.
            - int: Estimated communication cost in bytes.
    """

    weights_avg = deepcopy(weights[0])
    total_bytes = 0

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))

        # Count bytes transmitted for this parameter across all clients
        total_bytes += weights_avg[key].numel() * weights_avg[key].element_size() * len(weights)

    return weights_avg, total_bytes



"""
Implementation of FedACG aggregation. Adapted from the authors' reference code:
https://github.com/geehokim/FedACG

Original algorithm introduced in:
Kim, G., Kim, J. and Han, B. (2024)
'Communication-Efficient Federated Learning with Accelerated Client Gradient',
CVPR 2024. IEEE.
"""

def FedACG_lookahead(model, prev_global_state, acg_momentum):
    """
    Compute the FedACG lookahead model sent to clients.

    Args:
        model (nn.Module): Current global model.
        prev_global_state (dict): Previous global state_dict.
        acg_momentum (float): Momentum factor for accelerated gradients.

    Returns:
        nn.Module: Lookahead model incorporating accelerated updates.
    """

    lookahead_model = deepcopy(model)

    for (name, param) in lookahead_model.named_parameters():
        prev_param = prev_global_state[name]
        param.data += acg_momentum * (param.data - prev_param.data)

    return lookahead_model


def FedACG_aggregate(sending_model, clients_models):
    """
    Aggregate client updates in FedACG.

    Args:
        sending_model (nn.Module): Lookahead model sent to clients.
        clients_models (list[dict]): List of client state_dicts after local training.

    Returns:
        tuple:
            - dict: Updated global model weights.
            - int: Estimated communication cost in bytes.
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

    # Update global weights
    updated_weights = {k: sending_state[k] + aggregated_delta[k] for k in sending_state}

    # Estimate communication cost
    lookahead_bytes = sum(param.numel() * param.element_size() for param in sending_model.parameters())
    deltas_bytes = sum(sum(d.numel() * d.element_size() for d in delta.values()) for delta in clients_deltas)
    total_bytes = lookahead_bytes + deltas_bytes

    return updated_weights, total_bytes



"""
Implementation of LaS-FL aggregation. 
Adapted using the authors' reference code for Localize-and-Stitch:
https://github.com/uiuctml/Localize-and-Stitch

Original algorithm introduced in:
He, Y. et al. (2024) 
'Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic'. arXiv. 
Available at: https://doi.org/10.48550/arXiv.2408.13656.
"""

class Localiser:
    """
    Localiser component of LaS-FL, adapted from the Localize-and-Stitch approach.

    Computes sparse client updates relative to a pretrained global model.
    """
    def __init__(self, pretrained_state, sparsity=0.2):
        """
        Args:
            pretrained_state (dict): State_dict of pretrained global model.
            sparsity (float): Proportion of parameters to retain in deltas.
        """

        self.pretrained_state = pretrained_state
        self.sparsity = sparsity

    def compute_mask_and_deltas(self, client_state):
        """
        Compute sparse parameter updates and masks.

        Args:
            client_state (dict): Client model state_dict.

        Returns:
            tuple:
                - dict: Binary masks indicating transmitted parameters.
                - dict: Sparse deltas relative to pretrained_state.
                - int: Estimated communication cost (bytes).
        """

        mask, deltas, comm_cost = {}, {}, 0

        for name, w_pre in self.pretrained_state.items():
            if name not in client_state:
                continue
            w_client = client_state[name]
            delta = w_client - w_pre
            flat = delta.view(-1)
            k = max(1, int(self.sparsity * flat.numel()))

            # Top-k selection for sparse updates
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
            comm_cost += mask_tensor.sum().item() * 4  # float32 so 4 bytes

        return mask, deltas, comm_cost


class Stitcher:
    """
    Stitcher component of LaS-FL, adapted from the Localize-and-Stitch approach.

    Aggregates sparse client updates into a new global model.
    """
    def __init__(self, pretrained_state):
        """
        Args:
            pretrained_state (dict): Global model state_dict before stitching.
        """

        self.pretrained_state = deepcopy(pretrained_state)

    def stitch(self, client_deltas_list):
        """
        Combine sparse client deltas into a stitched global model.

        Args:
            client_deltas_list (list[dict]): List of client sparse deltas.

        Returns:
            dict: Updated global model state_dict after stitching.
        """

        stitched = deepcopy(self.pretrained_state)
        delta_accum = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in stitched.items()}
        counts = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in stitched.items()}

        # Accumulate deltas and counts
        for deltas in client_deltas_list:
            for name, d in deltas.items():
                if name not in delta_accum:
                    continue
                if not torch.is_floating_point(d):
                    continue  # skip non-float params
                mask = d.ne(0).to(d.device)
                delta_accum[name] += d
                counts[name] += mask.float()

        # Average non-zero updates across clients
        for name in stitched.keys():
            if not torch.is_floating_point(stitched[name]):
                continue # Skip ints
            nonzero = counts[name] > 0
            avg_delta = torch.zeros_like(delta_accum[name], dtype=torch.float32)
            avg_delta[nonzero] = delta_accum[name][nonzero] / counts[name][nonzero]
            stitched[name][nonzero] += avg_delta[nonzero]

        return stitched