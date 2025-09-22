import copy
import time
import torch

class Localiser:
    def __init__(self, sparsity):
        self.sparsity = sparsity

    def compute_importance_scores(self, base_model, trained_model):
        """
        Compute importance scores for each parameter tensor.
        Here: |delta| magnitude (simplified).
        """
        scores = []
        for p_base, p_trained in zip(base_model.parameters(), trained_model.parameters()):
            delta = p_trained.data - p_base.data
            scores.append(torch.abs(delta))
        return scores

    def localise(self, base_model, trained_model):
        """
        Produce (delta, mask) for a locally trained model.
        """
        delta, mask = [], []

        # deltas
        for p_base, p_trained in zip(base_model.parameters(), trained_model.parameters()):
            delta.append(p_trained.data - p_base.data)

        # scores + masks
        scores = self.compute_importance_scores(base_model, trained_model)
        for score_tensor in scores:
            k = max(1, int(self.sparsity * score_tensor.numel()))
            thresh = torch.topk(score_tensor.flatten(), k)[0][-1]
            mask_tensor = (score_tensor >= thresh).float()
            mask.append(mask_tensor.view_as(score_tensor))

        return delta, mask


class Stitcher:
    def __init__(self, conflict_rule="average"):
        self.conflict_rule = conflict_rule

    def stitch(self, base_model, deltas, masks):
        new_model = copy.deepcopy(base_model)

        with torch.no_grad():
            for param_idx, p_global in enumerate(base_model.parameters()):
                contribs = []
                for delta, mask in zip(deltas, masks):
                    contribs.append(delta[param_idx] * mask[param_idx])

                stacked = torch.stack(contribs, dim=0)
                selected = (stacked != 0).sum(0)

                if self.conflict_rule == "average":
                    merged_delta = torch.sum(stacked, dim=0)
                    merged_delta = torch.where(
                        selected > 0,
                        merged_delta / torch.clamp(selected, min=1),
                        merged_delta,
                    )
                elif self.conflict_rule == "sum":
                    merged_delta = torch.sum(stacked, dim=0)
                else:
                    raise ValueError("Unknown conflict rule")

                # inplace update
                list(new_model.parameters())[param_idx].data = (
                    p_global.data + merged_delta
                )

        return new_model