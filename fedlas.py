from datetime import datetime
import copy
import glob
import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import load_datasets, FederatedSampler
from models import ResNet
from utils import average_weights, FedACG_lookahead, FedACG_aggregate
from utils_las import Localiser, Stitcher


class FedAlg():
    """ Implementation based on code from the paper:
    McMahan, B. et al. (2017) 'Communication-Efficient Learning of Deep Networks from Decentralized Data', 
    in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics. Artificial Intelligence and Statistics, PMLR, pp. 1273-1282. 
    Available at: https://proceedings.mlr.press/v54/mcmahan17a.html.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            f"cuda:{0}" if torch.cuda.is_available() else "cpu"
        )

        self.dirichlet = self.args.dirichlet

        self.train_loader, self.val_loader, self.test_loader = self._get_data(
            root=self.args.data_root,
            n_clients=self.args.n_clients,
            dir_alpha=self.dirichlet
        )

        if self.args.dataset == "CIFAR10" or self.args.dataset == "CIFAR10T":
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

    def _get_data(self, root, n_clients, dir_alpha):
        """
        Args:
            root (str): path to the dataset.
            n_clients (int): number of clients.
            dir_alpha (float): Dirichlet distribution parameter.

        Returns:
            Tuple[DataLoader, DataLoader]: train_loader, test_loader
        """

        train_set, val_set, test_set = load_datasets(self.args.dataset)

        sampler = FederatedSampler(
            train_set, 
            n_clients=n_clients, 
            dir_alpha=dir_alpha)

        batch_size = self.args.batch_size

        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        test_loader = DataLoader(test_set, batch_size=batch_size)

        return train_loader, val_loader, test_loader

    def _train_client(self, root_model, train_loader, client_idx):
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

        n_batches = len(train_loader)

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
    
        return model, epoch_loss, epoch_acc
    
    def train(self):
        """Train a server model."""

        # Pattern to find existing folders for this experiment signature
        pattern = f"final baseline checkpoints/{self.args.algorithm}_{self.args.dataset}_{self.args.dirichlet}_seed{self.args.seed}_*"
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
                val_losses = ckpt["val_losses"]
                val_accs = ckpt["val_accs"]
                comm_bytes = ckpt.get("comm_bytes", [])
                train_times = ckpt.get("train_times", [])
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

            else:
                print("No checkpoint found in folder, starting from scratch")
                train_losses = []
                train_accs = []
                val_losses = []
                val_accs = []
                comm_bytes = []
                train_times = []
                start_epoch = 0
        else:
            exp_name = f"{self.args.algorithm}_{self.args.dataset}_{self.args.dirichlet}_seed{self.args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            checkpoint_dir = os.path.join("final baseline checkpoints", exp_name)
            os.makedirs(checkpoint_dir, exist_ok=True)

            print(f"Starting new experiment, checkpoints will be saved in: {checkpoint_dir}")
            train_losses = []
            train_accs = []
            val_losses = []
            val_accs = []
            comm_bytes = []
            train_times = []
            start_epoch = 0

        # Initialise previous global state for FedACG
        if self.args.algorithm == "fedacg":
            self.prev_global_state = copy.deepcopy(self.root_model.state_dict())

        # Initialise localiser and stitcher for las
        if self.args.algorithm == "las":
            localiser = Localiser(sparsity=self.args.sparsity)
            stitcher = Stitcher(conflict_rule="average")

        # Calculate number of clients to sample
        m = max(int(self.args.participation * self.args.n_clients), 1)

        # Start training
        for epoch in range(start_epoch, self.args.n_epochs):

            # Start timer for this round
            t1 = time.perf_counter()
            
            # Initialise general variables
            clients_models, clients_losses, clients_accs = [], [], []
            print(f"\nRound {epoch+1}/{self.args.n_epochs}")

            if self.args.algorithm == "las":
                # Initialise specific las variables
                deltas, masks, client_metrics = [], [], []
                total_bytes = 0

            if self.args.algorithm == "fedacg":
                global_model = FedACG_lookahead(
                    model=self.root_model,
                    prev_global_state=self.prev_global_state,
                    acg_momentum=self.args.acg_momentum
                )
            else:
                global_model = copy.deepcopy(self.root_model).to(self.device)
                global_model.eval()

            # Randomly select clients for this round
            idx_clients = np.random.choice(range(self.args.n_clients), m, replace=False)

            # Train individual clients with loading bar
            for client_idx in tqdm(idx_clients, desc="Training clients"):
                # Set client in the sampler
                self.train_loader.sampler.set_client(client_idx)

                # Train client locally
                client_model, client_loss, client_acc = self._train_client(
                    root_model=global_model,
                    train_loader=self.train_loader,
                    client_idx=client_idx,
                )

                # Append model, loss, and accuracy
                clients_models.append(client_model.state_dict())
                clients_losses.append(client_loss)
                clients_accs.append(client_acc)

                if self.args.algorithm == "las":
                    # Train mask and produce masked delta (localisation)
                    delta, mask = localiser.localise(global_model, client_model)
                    deltas.append(delta)
                    masks.append(mask)
                    total_bytes += 0 # local_bytes

            # Update server model based on clients models
            if self.args.algorithm in ["fedavg", "fedprox"]:
                updated_weights, total_bytes = average_weights(clients_models)

            elif self.args.algorithm == "fedacg":
                updated_weights, total_bytes = FedACG_aggregate(global_model, clients_models)
                self.prev_global_state = copy.deepcopy(updated_weights)

            elif self.args.algorithm == "las":
                # Stitch weights with global model to produce new global model
                updated_model = stitcher.stitch(global_model, deltas, masks)
                updated_weights = updated_model.state_dict()

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
            val_loss, val_acc = self.evaluate(split="val")
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # Print results
            print(f"\nResults after {epoch + 1} rounds of training:")
            print(f"---> Avg Train Loss: {train_loss:.4f} | Avg Train Accuracy: {train_acc*100:.4f}%")
            print(f"---> Avg Val Loss: {val_loss:.4f} | Avg Val Accuracy: {val_acc*100:.4f}%")
            print(f"---> Communication loss (bytes): {total_bytes}")

            comm_bytes.append(total_bytes)

            t2 = time.perf_counter()
            train_time = int(t2-t1)
            train_times.append(train_time)
            print(f"---> Training time: {train_time} seconds")

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
                    "train_times": train_times,
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
            
        self.avg_train_times = round(np.mean(train_times), 0)
        print(f"Average train time per epoch: {self.avg_train_times}seconds\n")

        self.avg_comm_bytes = round(np.mean(comm_bytes), 3)
        print(f"Average communication loss per epoch: {self.avg_comm_bytes}bytes\n")

        return train_losses, val_losses, train_accs, val_accs, checkpoint_dir

    def evaluate(self, split):
        """Evaluate the model on a held-out val/test set.

        Returns:
            Tuple[float, float]: average loss, average accuracy.
        """
        self.root_model.eval()

        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        if split == "test":
            all_preds = []
            all_targets = []
            loader = self.test_loader
        elif split == "val":
            loader = self.val_loader

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)

                logits = self.root_model(data)
                loss = F.cross_entropy(logits, target)

                total_loss += loss.item() * data.size(0)
                preds = logits.argmax(dim=1)
                total_correct += (preds == target).sum().item()
                total_samples += data.size(0)

                if split == "test":
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

        # calculate average accuracy and loss
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        # Compute extra metrics and log output in csv file
        if split == "test":
            # calculate precision, recall, f1
            precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

            # Print results to CLI
            print("Test results:")
            print(f"---> Avg Test Loss: {avg_loss:.4f} | Avg Test Accuracy: {avg_acc*100:.4f}%\n")

            # Log metrics for analysis
            results_path = "results.csv"
            results = {
                "method": self.args.algorithm,
                "dataset": self.args.dataset,
                "model": self.args.model_name,
                "Dirichlet": self.dirichlet,
                "rounds": self.args.n_epochs,
                "accuracy": avg_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "communication": self.avg_comm_bytes,
                "train_time": self.avg_train_times
            }

            pd.DataFrame([results]).to_csv(results_path,
                                            mode='a',
                                            header=not os.path.exists(results_path),
                                            index=False)
            
            print(f"\nResults saved to {results_path}")

        return avg_loss, avg_acc