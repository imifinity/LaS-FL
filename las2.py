import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import random
from copy import deepcopy
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from data import load_datasets, FederatedSampler
from models import ResNet


def main():
    set_seed(42)
    device = "cuda"
    
    # Get args
    args = arg_parser()
 
    dataset = args.dataset
    model_name = args.model_name
    dirichlet = args.dirichlet
    algorithm = args.algorithm
    
    print(algorithm + "\n" + dataset)
    print(dirichlet)

    train_loader, global_train_loader, val_loader, test_loader = _get_data(
        dataset = dataset,
        n_clients=args.n_clients,
        dir_alpha=dirichlet,
        batch_size = args.batch_size
    )

    if dataset == "CIFAR10" or dataset == "CIFAR10T":
        n_classes = 10

        if model_name == "resnet18":
            root_model = ResNet(depth=18, n_classes=n_classes).to(device)
        elif model_name == "resnet50":
            root_model = ResNet(depth=50, n_classes=n_classes).to(device)
        else:
            raise ValueError(f"Invalid model name, {model_name}")

    elif dataset == "TinyImageNet":
        n_classes = 200

        if model_name == "resnet18":
            root_model = ResNet(depth=18, n_classes=n_classes, pretrained=True).to(device)
        elif model_name == "resnet50":
            root_model = ResNet(depth=50, n_classes=n_classes, pretrained=True).to(device)
        else:
            raise ValueError(f"Invalid model name, {model_name}")

    else:
        raise ValueError(f"Invalid dataset name, {dataset}")

    global_model, metrics, test_acc, f1, precision, recall = federated_training(
        root_model,
        algorithm,
        train_loader,
        global_train_loader,
        val_loader,
        test_loader,
        n_clients=args.n_clients,
        participation=args.participation,
        rounds=args.n_epochs,
        local_epochs=args.n_client_epochs,
        sparsity=args.sparsity,
        device="cuda"
    )

    # Log performance metrics
    os.makedirs("xmetrics", exist_ok=True)
    metrics_path = os.path.join("xmetrics", f"{algorithm}_{dataset}_{dirichlet}_{args.seed}_metrics.csv")
    results = {
        "epoch": np.arange(1, args.n_epochs+1),
        "train_losses": metrics["train_loss"],
        "train_accs": metrics["train_acc"],
        "val_losses": metrics["val_loss"],
        "val_accs": metrics["val_acc"]
    }

    pd.DataFrame([results]).to_csv(metrics_path,
                                    mode='a',
                                    header=not os.path.exists(metrics_path),
                                    index=False)

    print(f"\nMetrics saved to {metrics_path}")

    avg_comm_bytes = np.mean(metrics["comm_costs"])
    avg_train_times = np.mean(metrics["train_times"])
    
    # Log results
    results_path = "xresults.csv"
    results = {
        "method": algorithm,
        "dataset": dataset,
        "Dirichlet": dirichlet,
        "rounds": args.n_epochs,
        "accuracy": test_acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "communication": avg_comm_bytes,
        "train_time": avg_train_times
    }

    pd.DataFrame([results]).to_csv(results_path,
                                    mode='a',
                                    header=not os.path.exists(results_path),
                                    index=False)
    
    print(f"Results saved to {results_path}")

    # Plot curves
    os.makedirs("xplots", exist_ok=True)
    path1 = plot_curve(algorithm, dirichlet, args.seed, "accuracy", train_accs, val_accs, "xplots")
    path2 = plot_curve(algorithm, dirichlet, "loss", train_losses, val_losses, "xplots")
    print(f"Results saved to {path1} and {path2}")


def log(*args, **kwargs):
    """Always flush (print) output immediately"""
    print(*args, **kwargs, flush=True)


def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def arg_parser() -> argparse.ArgumentParser:
    """Collect command-line arguments"""
    parser = argparse.ArgumentParser()

    # Mode args
    parser.add_argument("--data_root", type=str, default="./data/")
    parser.add_argument("--algorithm", type=str, default="fedavg")
    parser.add_argument("--dataset", type=str, default="CIFAR10")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--seed", type=int, default=0)

    # Federation args
    parser.add_argument("--dirichlet", type=float, default=np.nan)
    parser.add_argument("--n_clients", type=int, default=20)
    parser.add_argument("--participation", type=float, default=0.2)

    # Training args
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_client_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)

    # Aggregation method args 
    parser.add_argument('--sparsity', type=float, default=0.2)
    parser.add_argument("--prox_momentum", type=float, default=0.9)
    parser.add_argument("--acg_momentum", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)

    return parser.parse_args()


def plot_curve(algorithm, dirichlet, seed, metric, train, val, filepath):

    # Plot curves
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train) + 1), train, marker='o', label='train')
    plt.plot(range(1, len(val) + 1), val, marker='o', label='validation')
    plt.title(f"{metric.title()} Curves")
    plt.xlabel("Round (Epoch)")
    plt.ylabel(metric.title())
    plt.grid(True)
    plt.legend()
    
    # Save plot as PNG file
    path = os.path.join(filepath, f"{algorithm}_{dirichlet}_{seed}_{metric}_curves.png")
    plt.savefig(path)

    return path


def _get_data(dataset, n_clients, dir_alpha, batch_size):
    """
    Args:
        root (str): path to the dataset.
        n_clients (int): number of clients.
        dir_alpha (float): Dirichlet distribution parameter.

    Returns:
        Tuple[DataLoader, DataLoader]: train_loader, test_loader
    """

    train_set, val_set, test_set = load_datasets(dataset)

    sampler = FederatedSampler(
        train_set, 
        n_clients=n_clients, 
        dir_alpha=dir_alpha)

    batch_size = batch_size

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    global_train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    return train_loader, global_train_loader, val_loader, test_loader


# ---------------- Localiser ---------------- #
class Localiser:
    def __init__(self, pretrained_state, sparsity=0.2):
        self.pretrained_state = pretrained_state
        self.sparsity = sparsity

    def compute_mask_and_deltas(self, client_state):
        mask, deltas = {}, {}
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
        return mask, deltas


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


# ---------------- average_weights ---------------- #
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

# ---------------- Client Update ---------------- #
def client_update(model, train_loader, epochs, device):
    model = deepcopy(model).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    return model.state_dict()


# ---------------- Evaluation ---------------- #
def evaluate(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return acc, avg_loss


def evaluate(model, data_loader, device, split):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    
    if split == "test":
        all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            if split == "test":
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / total
    acc = 100.0 * correct / total

    if split == "test":
        # Compute precision, recall, F1 (macro average = treats all classes equally)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        
        return acc, avg_loss, f1, precision, recall

    else:
        return acc, avg_loss


# ---------------- Federated Training Loop ---------------- #
def federated_training(global_model, algorithm, train_loader, global_train_loader, 
                        val_loader, test_loader, n_clients, rounds, 
                        participation, local_epochs, sparsity, device):

    global_model.to(device)
    pretrained_state = deepcopy(global_model.state_dict())

    m = max(int(participation * n_clients), 1)

    metrics = {
        "train_acc": [],
        "train_loss": [],
        "val_acc": [],
        "val_loss": [],
        "comm_costs": [],
        "train_times": []
    }

    for r in range(rounds):
        t1 = time.perf_counter()
        print(f"\nRound {r+1}/{rounds}")

        idx_clients = np.random.choice(range(n_clients), m, replace=False)
        client_deltas_list, comm = [], 0

        print(f"\nTraining clients", end="")
        for client_idx in idx_clients:
            print(".", end="")
            
            # Point sampler to this client
            train_loader.sampler.set_client(int(client_idx))
            # Train client locally
            client_state = client_update(
                global_model, train_loader, local_epochs, device
            )

            if algorithm == "las":
                # Localise deltas
                localiser = Localiser(pretrained_state, sparsity)
                _, deltas = localiser.compute_mask_and_deltas(client_state)
                client_deltas_list.append(deltas)
            elif algorithm in ["fedavg", "fedprox"]:
                # Collect full client states
                client_deltas_list.append(client_state)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        if algorithm == "las":
            # Stitch results into new global state
            stitcher = Stitcher(pretrained_state)
            new_global_state = stitcher.stitch(client_deltas_list)
        elif algorithm == "fedavg":
            # Average weights
            new_global_state, comm = average_weights(client_deltas_list)

        metrics["comm_costs"].append(comm)

        # Update global model
        global_model.load_state_dict(new_global_state)
        pretrained_state = deepcopy(new_global_state)

        # Evaluate using train and validation sets
        train_acc, train_loss = evaluate(global_model, global_train_loader, device, split="train")
        val_acc, val_loss = evaluate(global_model, val_loader, device, split="val")

        # Print results
        print(f"\nResults after {r + 1} rounds of training:")
        print(f"---> Avg Train Loss: {train_loss:.4f} | Avg Train Accuracy: {train_acc:.4f}%")
        print(f"---> Avg Val Loss: {val_loss:.4f} | Avg Val Accuracy: {val_acc:.4f}%")
        print(f"---> Communication loss (bytes): {comm}")

        # Append metrics to dict
        metrics["train_acc"].append(train_acc)
        metrics["train_loss"].append(train_loss)
        metrics["val_acc"].append(val_acc)
        metrics["val_loss"].append(val_loss)
        
        # Get training time
        t2 = time.perf_counter()
        train_time = int(t2-t1)
        metrics["train_times"].append(train_time)
        print(f"---> Training time: {train_time} seconds")

    # Final test
    test_acc, _, f1, precision, recall = evaluate(global_model, test_loader, device, split="test")
    print("\nTest results:")
    print(f"---> Accuracy: {test_acc:.4f}%\n")
    print(f"---> F1 Score: {f1:.4f}%\n")
    print(f"---> Precision: {precision:.4f}%\n")
    print(f"---> Recall: {recall:.4f}%\n")

    return global_model, metrics, test_acc, f1, precision, recall


if __name__ == "__main__":
    main()