import argparse
from datetime import datetime
import copy
import glob
import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple

from data import load_datasets, FederatedSampler
from models import ResNet
from utils import arg_parser, average_weights, FedACG_lookahead, FedACG_aggregate, FedDyn_aggregate, Localiser, Stitcher


class FedAlg():
    """ Implementation based on code from the paper:
    McMahan, B. et al. (2017) 'Communication-Efficient Learning of Deep Networks from Decentralized Data', 
    in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics. Artificial Intelligence and Statistics, PMLR, pp. 1273-1282. 
    Available at: https://proceedings.mlr.press/v54/mcmahan17a.html.
    """

    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        )

        self.train_loader, self.val_loader, self.test_loader = self._get_data(
            root=self.args.data_root,
            n_clients=self.args.n_clients,
            dir_alpha=self.args.dirichlet,
            non_iid=self.args.non_iid,
        )

        if self.args.dataset == "CIFAR10":
            self.n_classes = 10

            if self.args.model_name == "resnet18":
                self.root_model = ResNet(depth=18, n_classes=self.n_classes).to(self.device)
            elif self.args.model_name == "resnet50":
                self.root_model = ResNet(depth=50, n_classes=self.n_classes).to(self.device)
            else:
                raise ValueError(f"Invalid model name, {self.args.model_name}")

        elif self.args.dataset == "TinyImageNet":
            self.n_classes = 200

            if self.args.model_name == "resnet18":
                self.root_model = ResNet(depth=18, n_classes=self.n_classes, pretrained=True).to(self.device)
            elif self.args.model_name == "resnet50":
                self.root_model = ResNet(depth=50, n_classes=self.n_classes, pretrained=True).to(self.device)
            else:
                raise ValueError(f"Invalid model name, {self.args.model_name}")

        else:
            raise ValueError(f"Invalid dataset name, {self.args.dataset}")

        self.target_acc = 0.99

        self.graft_args = argparse.Namespace(
            device=self.args.device,
            topk=self.args.topk,
            graft_lr=self.args.graft_lr,
            graft_epochs=self.args.graft_epochs,
            sparsity=self.args.sparsity,
            sigmoid_bias=self.args.sigmoid_bias,
            l1_strength=self.args.l1_strength
        )

        self.reached_target_at = None  # type: int
    
    def _flatten_state_dict(self, state_dict):
        # Flatten a state_dict into a 1D tensor with consistent ordering
        return torch.cat([param.view(-1) for _, param in state_dict.items()])

    def _get_data(
        self, root: str, n_clients: int, dir_alpha: float, non_iid: int
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Args:
            root (str): path to the dataset.
            n_clients (int): number of clients.
            dir_alpha (float): Dirichlet distribution parameter.
            non_iid (int): 0: IID, 1: Non-IID

        Returns:
            Tuple[DataLoader, DataLoader]: train_loader, test_loader
        """

        train_set, val_set, test_set = load_datasets(self.args.dataset)

        sampler = FederatedSampler(train_set, non_iid=non_iid, n_clients=n_clients, dir_alpha=dir_alpha)

        batch_size = self.args.batch_size

        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)

        return train_loader, val_loader, test_loader

    def _train_client(
        self, root_model: nn.Module, train_loader: DataLoader, client_idx: int
    ) -> Tuple[nn.Module, float]:
        """Train a client model.

        Args:
            root_model (nn.Module): server model.
            train_loader (DataLoader): client data loader.
            client_idx (int): client index.

        Returns:
            Tuple[nn.Module, float]: client model, average client loss.
        """

        model = copy.deepcopy(root_model)
        model.train()
        
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=self.args.momentum
        )

        n_batches = len(self.train_loader)

        # Store global params for FedProx
        if self.args.algorithm == "fedprox":
            global_params = {k: v.clone().detach().to(self.device) for k, v in root_model.state_dict().items()} 

        for epoch in range(self.args.n_client_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                logits = model(data)
                loss = F.cross_entropy(logits, target)

                # FedDyn - dynamic regularisation
                """ Code based on the paper:
                Acar, D.A.E. et al. (2021) 'Federated Learning Based on Dynamic Regularization'. arXiv. 
                Available at: https://doi.org/10.48550/arXiv.2111.04263.
                """
                if self.args.algorithm == "feddyn":
                    # Flatten client model, server model, and history using state_dict

                    p_vec = self._flatten_state_dict(model.state_dict())
                    s_vec = self._flatten_state_dict(root_model.state_dict())
                    h_vec = self._flatten_state_dict(self.histories[client_idx])

                    dynamic_loss = (
                        0.5 * self.args.alpha * (p_vec**2).sum()
                        - self.args.alpha * (p_vec * (s_vec - h_vec)).sum()
                    )
                    loss = loss + dynamic_loss

                # FedProx - add proximal term
                """ Code based on the paper:
                Yuan, X.-T. and Li, P. (2022) 'On Convergence of FedProx: Local Dissimilarity Invariant Bounds, 
                Non-smoothness and Beyond', Advances in Neural Information Processing Systems, 35, pp. 10752-10765.
                """

                if self.args.algorithm == "fedprox":
                    prox_loss = 0.0
                    for (name, param) in model.named_parameters():
                        prox_loss += ((param - global_params[name].to(param.device)) ** 2).sum()
                    loss = loss + (self.args.prox_momentum / 2.0) * prox_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * data.size(0)
                epoch_correct += (logits.argmax(dim=1) == target).sum().item()
                epoch_samples += data.size(0)

            # Calculate average accuracy and loss
            epoch_loss /= epoch_samples
            epoch_acc = epoch_correct / epoch_samples

            print(f"Client #{client_idx} | Epoch: {epoch+1}/{self.args.n_client_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.4f}%")
            
        return model, epoch_loss, epoch_acc

    def train(self) -> None:
        """Train a server model."""
        
        t0 = time.perf_counter()

        # Initialise previous global state for FedACG
        if self.args.algorithm == "fedacg":
            self.prev_global_state = copy.deepcopy(self.root_model.state_dict())

        # Initialise FedDyn histories
        if self.args.algorithm == "feddyn":
            # Each client's history corresponds to a correction vector
            sd = self.root_model.state_dict()
            self.histories = [
                {k: torch.zeros(v.size(), dtype=torch.float32, device=self.device) for k, v in sd.items()}
                for _ in range(self.args.n_clients)
            ]

        # Pattern to find existing folders for this experiment signature
        pattern = f"checkpoints/{self.args.algorithm}_{self.args.dataset}_seed{self.args.seed}_*"
        existing_folders = glob.glob(pattern)

        if existing_folders:
            # Resume from the most recent experiment folder
            checkpoint_dir = max(existing_folders, key=os.path.getctime)
            print(f"Found existing experiment folder: {checkpoint_dir}")

            # Look for checkpoint files in that folder
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch*.pt"))
            if checkpoint_files:
                latest_ckpt = max(checkpoint_files, key=os.path.getctime)
                ckpt = torch.load(latest_ckpt, map_location=self.device)
                print(f"Resuming from checkpoint: {latest_ckpt}")

                # Load server model
                self.root_model.load_state_dict(ckpt["model_state_dict"])

                # Load other state
                train_losses = ckpt["train_losses"]
                train_accs = ckpt["train_accs"]
                #val_losses = ckpt["val_losses"]
                #val_accs = ckpt["val_accs"]
                comm_bytes = ckpt.get("comm_bytes", [])
                self.prev_global_state = ckpt.get("prev_global_state", None)
                if hasattr(self, "optimizer") and ckpt.get("optimizer_state_dict") is not None:
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt["epoch"]

                # Trim csv file so it only has rows <= start_epoch
                metrics_path = os.path.join(checkpoint_dir, f"{self.args.algorithm}_metrics.csv")
                if os.path.exists(metrics_path):
                    df = pd.read_csv(metrics_path)
                    df = df[df["epoch"] <= start_epoch]
                    df.to_csv(metrics_path, index=False)
                    print(f"Trimmed metrics.csv to {len(df)} rows (up to epoch {start_epoch})")

                val_losses = []
                val_accs = []

            else:
                # No checkpoint in folder → start fresh
                print("No checkpoint found in folder, starting from scratch")
                train_losses = []
                train_accs = []
                val_losses = []
                val_accs = []
                comm_bytes = []
                start_epoch = 0
        else:
            # No existing folder → start fresh
            exp_name = f"{self.args.algorithm}_{self.args.dataset}_seed{self.args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            checkpoint_dir = os.path.join("checkpoints", exp_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Starting new experiment, checkpoints will be saved in: {checkpoint_dir}")
            
            train_losses = []
            train_accs = []
            val_losses = []
            val_accs = []
            comm_bytes = []
            start_epoch = 0

        for epoch in range(start_epoch, self.args.n_epochs):
            t1 = time.perf_counter()

            print(f"\nRound {epoch+1}/{self.args.n_epochs}\n")

            clients_models = []
            clients_losses = []
            clients_accs = []

            # Randomly select clients
            m = max(int(self.args.participation * self.args.n_clients), 1)
            idx_clients = np.random.choice(range(self.args.n_clients), m, replace=False)

            if self.args.algorithm == "fedacg":
                sending_model = FedACG_lookahead(
                    model=self.root_model,
                    prev_global_state=self.prev_global_state,
                    acg_momentum=self.args.acg_momentum
                )
            else:
                sending_model = copy.deepcopy(self.root_model).to(self.device)
                sending_model.eval()

            if self.args.algorithm == "las":

                ##### Testing #####
                print("=== Round start ===")
                print("Sending model device:", next(sending_model.parameters()).device)
                print("Train loader batches:", len(self.train_loader))
                round_time_start = time.time()
                ##### Testing #####

                masks = []
                ft_models = []
                total_bytes = 0

                # Make a copy of the model before training
                self.pretrained_model = copy.deepcopy(sending_model)

                # Train clients
                sending_model.train()

                for client_idx in idx_clients:
                    # Set client in the sampler
                    self.train_loader.sampler.set_client(client_idx)

                    # BEFORE training this client
                    before_root = {n: p.detach().cpu().clone() for n, p in sending_model.named_parameters()}

                    # local debug: record a couple of scalar summaries of the root model
                    first_name = list(before_root.keys())[0]
                    last_name = list(before_root.keys())[-1]
                    print(f"Before training client {client_idx}: root first param sum={before_root[first_name].sum().item():.6e}, "
                        f"root last param sum={before_root[last_name].sum().item():.6e}")

                    t_client_start = time.perf_counter()

                    # Train client
                    client_model, client_loss, client_acc = self._train_client(
                        root_model=sending_model,
                        train_loader=self.train_loader,
                        client_idx=client_idx,
                    )

                    t_client_end = time.perf_counter()

                    # AFTER training: compare root -> client
                    after_client = {n: p.detach().cpu().clone() for n, p in client_model.named_parameters()}

                    # compute L1 difference between root(before) and client(after)
                    param_l1_diff_root_to_client = 0.0
                    for n in before_root.keys():
                        if n in after_client:
                            param_l1_diff_root_to_client += (before_root[n] - after_client[n]).abs().sum().item()

                    # Also compute L1 difference between client after and itself re-loaded (sanity check)
                    # and small per-layer sums to see where changes happen
                    first_after = after_client[first_name].sum().item()
                    last_after = after_client[last_name].sum().item()

                    print(f"Client {client_idx} train wall-time: {(t_client_end - t_client_start):.4f}s")
                    print(f"Client {client_idx} param L1 diff (root_before -> client_after): {param_l1_diff_root_to_client:.6e}")
                    print(f"Client {client_idx} returned loss: {client_loss:.6e}, client first param sum={first_after:.6e}, client last param sum={last_after:.6e}")


                    clients_models.append(client_model.state_dict())
                    clients_losses.append(client_loss)
                    clients_accs.append(client_acc)
                    ft_models.append(client_model)

                    localiser = Localiser(
                            trainable_params=dict(client_model.named_parameters()),
                            model=sending_model,
                            pretrained_model=self.pretrained_model,
                            finetuned_model=client_model,
                            graft_args=self.graft_args
                        )

                    mask, proportion, acc, bytes_this_client = localiser.train_graft(self.train_loader)
                    masks.append(mask)
                    total_bytes += bytes_this_client

                stitcher = Stitcher(
                    trainable_params=dict(client_model.named_parameters()),
                    pretrained_model=self.pretrained_model,
                    finetuned_models=ft_models,
                    masks=masks)

                stitched_model, bytes_stitcher = stitcher.interpolate_models()
                stitched_model.to(self.device)

                for name, param in stitched_model.named_parameters():
                    print(f"{name}: {param.norm().item():.6f}")

                total_bytes += bytes_stitcher
                updated_weights = stitched_model.state_dict()

                ##### Testing #####
                round_time_end = time.time()
                print(f"Round time (s): {round_time_end - round_time_start:.4f}")
                # quick param-norm of global model
                global_norm = sum(p.data.norm().item() for p in sending_model.parameters())
                print("Global model param norm:", global_norm)
                ##### Testing #####

            else:
                for client_idx in idx_clients:
                    self.train_loader.sampler.set_client(client_idx)

                    # Train client
                    client_model, client_loss, client_acc = self._train_client(
                        root_model=sending_model,
                        train_loader=self.train_loader,
                        client_idx=client_idx,
                    )
                    clients_models.append(client_model.state_dict())
                    clients_losses.append(client_loss)
                    clients_accs.append(client_acc)

                # Update server model based on clients models
                if self.args.algorithm in ["fedavg", "fedprox"]:
                    updated_weights, total_bytes = average_weights(clients_models)

                elif self.args.algorithm == "feddyn":
                    updated_weights, self.histories, total_bytes = FedDyn_aggregate(
                        local_weights=clients_models,
                        global_weights=self.root_model.state_dict(),
                        histories=self.histories,
                        selected_clients=idx_clients,
                        alpha=self.args.alpha
                    )

                elif self.args.algorithm == "fedacg":
                    updated_weights, total_bytes = FedACG_aggregate(sending_model, clients_models)
                    self.prev_global_state = copy.deepcopy(updated_weights)

                else:
                    raise ValueError(f"Invalid algorithm name, {self.args.algorithm}")

            # Update the server model
            self.root_model.to(self.device)
            self.root_model.load_state_dict(updated_weights)

            # Update average loss of this round
            train_loss = sum(clients_losses) / len(clients_losses)
            train_losses.append(train_loss)

            # Update average accs of this round
            train_acc = sum(clients_accs) / len(clients_accs)
            train_accs.append(train_acc)

            # Validate server model
            val_loss, val_acc = self.validate()
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # Print results
            print(f"\nResults after {epoch + 1} rounds of training:")
            print(f"---> Avg Train Loss: {train_loss:.4f} | Avg Train Accuracy: {train_acc*100:.4f}%")
            print(f"---> Avg Val Loss: {val_loss:.4f} | Avg Val Accuracy: {val_acc*100:.4f}%")
            print(f"---> Communication loss (bytes): {total_bytes}")

            comm_bytes.append(total_bytes)

            t2 = time.perf_counter()
            print(f"---> Training time: {(t2-t1):.3f} seconds")

            # Log every n epochs
            n = self.args.log
            if (epoch + 1) % n == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.root_model.state_dict(),
                    "train_losses": train_losses,
                    "train_accs": train_accs,
                    "val_losses": val_losses,
                    "val_accs": val_accs,
                    "comm_bytes": comm_bytes,
                    "optimizer_state_dict": getattr(self, "optimizer", None),
                    "prev_global_state": getattr(self, "prev_global_state", None),
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt"))
                print(f"\nCheckpoint saved at epoch {epoch+1}")

            # Log metrics for analysis
            metrics_path = os.path.join(checkpoint_dir, f"{self.args.algorithm}_metrics.csv")
            results = {
                "epoch": epoch + 1,
                "train_losses": train_loss,
                "train_accs": train_acc,
                "val_losses": val_loss,
                "val_accs": val_acc
            }

            pd.DataFrame([results]).to_csv(metrics_path,
                                            mode='a',
                                            header=not os.path.exists(metrics_path),
                                            index=False)
            
        t3 = time.perf_counter()

        self.train_time = round(t3 - t0, 3)
        print(f"\nTraining took {self.train_time}seconds\n")
       
        self.avg_comm_bytes = round(np.mean(comm_bytes), 3)
        print(f"Average communication loss per epoch: {self.avg_comm_bytes}bytes\n")

        return train_losses, val_losses, train_accs, val_accs, checkpoint_dir
        
    def validate(self) -> Tuple[float, float]:
        """Validate the server model.

        Returns:
            Tuple[float, float]: average loss, average accuracy.
        """
        self.root_model.eval()

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)

                logits = self.root_model(data)
                loss = F.cross_entropy(logits, target)

                total_loss += loss.item() * data.size(0)
                total_correct += (logits.argmax(dim=1) == target).sum().item()
                total_samples += data.size(0)

        # calculate average accuracy and loss
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def test(self) -> Tuple[float, float]:
        """Test the final model on the held-out test set.

        Returns:
            Tuple[float, float]: average loss, average accuracy.
        """
        self.root_model.eval()

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                logits = self.root_model(data)
                loss = F.cross_entropy(logits, target)

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                total_correct += (preds == target).sum().item()
                total_samples += data.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # calculate average accuracy and loss
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        # calculate precision, recall, f1
        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')

        # Print results to CLI
        print("Test results:")
        print(f"---> Avg Test Loss: {avg_loss:.4f} | Avg Test Accuracy: {avg_acc*100:.4f}%\n")

        # Log metrics for analysis
        results_path = "more_results.csv"
        results = {
            "method": self.args.algorithm,
            "dataset": self.args.dataset,
            "model": self.args.model_name,
            "IID": self.args.non_iid,
            "rounds": self.args.n_epochs,
            "accuracy": avg_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "communication": self.avg_comm_bytes,
            "train_time": self.train_time
        }

        pd.DataFrame([results]).to_csv(results_path,
                                        mode='a',
                                        header=not os.path.exists(results_path),
                                        index=False)
        
        print(f"\nResults saved to {results_path}")