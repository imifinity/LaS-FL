import copy
import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Optional, Tuple
from utils import arg_parser

import argparse
from data import load_datasets, FederatedSampler
from models import ResNet
from utils import average_weights, Localiser, Stitcher, FedACG_lookahead


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

            if self.args.model_name == "cnn":
                self.root_model = CNN().to(self.device)
            elif self.args.model_name == "resnet18":
                self.root_model = ResNet(depth=18, n_classes=10).to(self.device)
            elif self.args.model_name == "resnet50":
                self.root_model = ResNet(depth=50, n_classes=10).to(self.device)
            else:
                raise ValueError(f"Invalid model name, {self.args.model_name}")

        elif self.args.dataset == "TinyImageNet":

            if self.args.model_name == "resnet18":
                self.root_model = ResNet(depth=18, n_classes=200, pretrained=True).to(self.device)
            elif self.args.model_name == "resnet50":
                self.root_model = ResNet(depth=50, n_classes=200, pretrained=True).to(self.device)
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
        
        # FedDyn - initialise alpha
        if self.args.algorithm == "feddyn":
            if not hasattr(self, "client_alphas"):
                self.client_alphas = [None] * self.args.n_clients
            if self.client_alphas[client_idx] is None:
                self.client_alphas[client_idx] = [torch.zeros_like(p) for p in model.parameters()]
            alpha = self.client_alphas[client_idx]
        
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=self.args.momentum
        )

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
                Jin, C. et al. (2023) 'FedDyn: A dynamic and efficient federated distillation approach on Recommender 
                System', in 2022 IEEE 28th International Conference on Parallel and Distributed Systems (ICPADS). 2022 
                IEEE 28th International Conference on Parallel and Distributed Systems (ICPADS), pp. 786-793. 
                Available at: https://doi.org/10.1109/ICPADS56603.2022.00107.
                """
                if self.args.algorithm == "feddyn":
                    # Dynamic term: <theta, alpha>
                    dynamic_loss = sum((p * a).sum() for p, a in zip(model.parameters(), alpha))
                    
                    # Optional proximal term (like FedProx)
                    mu = getattr(self.args, "mu", 0.0)
                    if mu > 0:
                        global_params = [p.clone().detach() for p in root_model.parameters()]
                        prox_loss = (mu / 2) * sum(((p - p0)**2).sum() for p, p0 in zip(model.parameters(), global_params))
                    else:
                        prox_loss = 0.0
                    
                    loss = loss + dynamic_loss + prox_loss
                
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

                epoch_loss += loss.item()
                epoch_correct += (logits.argmax(dim=1) == target).sum().item()
                epoch_samples += data.size(0)

            # Calculate average accuracy and loss
            epoch_loss /= idx
            epoch_acc = epoch_correct / epoch_samples

            # FedDyn - update alpha
            if self.args.algorithm == "feddyn":
                with torch.no_grad():
                    for i, p in enumerate(model.parameters()):
                        alpha[i] = alpha[i] - (p - list(root_model.parameters())[i]) / self.args.n_clients

            print(f"Client #{client_idx} | Epoch: {epoch+1}/{self.args.n_client_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.4f}%")
            
        return model, epoch_loss / self.args.n_client_epochs

    def train(self) -> None:
        """Train a server model."""
        train_losses = []
        
        t0 = time.perf_counter()

        # Initialize previous global state for FedACG
        if self.args.algorithm == "fedacg":
            self.prev_global_state = copy.deepcopy(self.root_model.state_dict())

        comm_bytes = []

        for epoch in range(self.args.n_epochs):
            t1 = time.perf_counter()

            print(f"\nRound {epoch+1}/{self.args.n_epochs}\n")

            clients_models = []
            clients_losses = []

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
                sending_model = self.root_model

            if self.args.algorithm == "las":
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

                    # Train client
                    client_model, client_loss = self._train_client(
                        root_model=sending_model,
                        train_loader=self.train_loader,
                        client_idx=client_idx,
                    )

                    clients_models.append(client_model.state_dict())
                    clients_losses.append(client_loss)
                    ft_models.append(client_model)

                    localiser = Localiser(
                            trainable_params=dict(client_model.named_parameters()),
                            model=sending_model.eval(),
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
                total_bytes += bytes_stitcher

                updated_weights = stitched_model.state_dict()

            else:
                for client_idx in idx_clients:
                    self.train_loader.sampler.set_client(client_idx)

                    # Train client
                    client_model, client_loss = self._train_client(
                        root_model=sending_model,
                        train_loader=self.train_loader,
                        client_idx=client_idx,
                    )
                    clients_models.append(client_model.state_dict())
                    clients_losses.append(client_loss)

                # Update server model based on clients models
                if self.args.algorithm in ["fedavg", "fedprox", "fedacg", "feddyn"]:
                    updated_weights, total_bytes = average_weights(clients_models)
                else:
                    raise ValueError(f"Invalid algorithm name, {self.args.algorithm}")

            # Update the server model
            self.root_model.load_state_dict(updated_weights)

            # FedAcg only - compute local deltas relative to what clients received
            if self.args.algorithm == "fedacg":
                clients_deltas = []
                sending_state = sending_model.state_dict()
                for client_state in clients_models:
                    delta = {k: client_state[k] - sending_state[k] for k in sending_state}
                    clients_deltas.append(delta)
                # Update previous global state for next FedACG lookahead
                self.prev_global_state = copy.deepcopy(self.root_model.state_dict())

            # Update average loss of this round
            avg_loss = sum(clients_losses) / len(clients_losses)
            train_losses.append(avg_loss)

            if (epoch + 1) % self.args.log_every == 0:
                # Test server model
                total_loss, total_acc = self.validate()
                avg_train_loss = sum(train_losses) / len(train_losses)

                # Log results
                logs = {
                    "train/loss": avg_train_loss,
                    "val/loss": total_loss,
                    "val/acc": total_acc,
                    "round": epoch,
                }
                if total_acc >= self.target_acc and self.reached_target_at is None:
                    self.reached_target_at = epoch
                    logs["reached_target_at"] = self.reached_target_at
                    print(f"\n -----> Target accuracy {self.target_acc*100}% reached at round {epoch}! <----- \n")

                # Print results
                print(f"\nResults after {epoch + 1} rounds of training:")
                print(f"---> Avg Training Loss: {avg_train_loss:.4f}")
                print(f"---> Avg Val Loss: {total_loss:.4f} | Avg Val Accuracy: {total_acc*100:.4f}%")
                print(f"Communication loss (bytes): {total_bytes}")

                comm_bytes.append(total_bytes)

                t2 = time.perf_counter()
                print(f"Round {epoch+1} took {(t2-t1):.3f} seconds")

                # Early stopping
                if self.args.early_stopping and self.reached_target_at is not None:
                    print(f"\nEarly stopping at round #{epoch+1}...")
                    break

        t3 = time.perf_counter()

        self.train_time = round(t3 - t0, 3)
        print(f"\nTraining took {self.train_time}seconds\n")
       
        self.avg_comm_bytes = round(np.mean(comm_bytes), 3)
        print(f"Average communication loss per epoch: {self.avg_comm_bytes}bytes\n")
    
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
            for idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)

                logits = self.root_model(data)
                loss = F.cross_entropy(logits, target)

                total_loss += loss.item()
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

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                logits = self.root_model(data)
                loss = F.cross_entropy(logits, target)

                total_loss += loss.item()
                total_correct += (logits.argmax(dim=1) == target).sum().item()
                total_samples += data.size(0)

        # calculate average accuracy and loss
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        # Log results
        logs = {
            "test/loss": avg_loss,
            "test/acc": avg_acc
        }

        # Print results to CLI
        print("Test results:")
        print(f"---> Avg Test Loss: {avg_loss:.4f} | Avg Test Accuracy: {avg_acc*100:.4f}%\n")

        # Write results to CSV file
        results = {
            "method": self.args.algorithm,
            "dataset": self.args.dataset,
            "model": self.args.model_name,
            "rounds": self.args.n_epochs,
            "accuracy": avg_acc,
            "loss": avg_loss,
            "communication": self.avg_comm_bytes,
            "train_time": self.train_time
        }

        # Convert to dataframe
        results_df = pd.DataFrame([results])

        # File to save results
        r_filename = "experiment_results.csv"

        # Append if file exists, otherwise create new file with header
        if os.path.isfile(r_filename):
            df.to_csv(r_filename, mode='a', header=False, index=False)
        else:
            df.to_csv(r_filename, index=False)