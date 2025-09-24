import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from copy import deepcopy

from data import load_datasets, FederatedSampler
from models import ResNet
from utils import set_seed, arg_parser, plot_curve
from agg_utils import average_weights, FedACG_lookahead, FedACG_aggregate, Localiser, Stitcher


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

    train_loader, global_train_loader, val_loader, test_loader = get_data(
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
        prox_mu=args.prox_momentum,
        local_epochs=args.n_client_epochs,
        sparsity=args.sparsity,
        device="cuda",
        acg_momentum=args.acg_momentum
    )

    # Log performance metrics
    os.makedirs("metrics", exist_ok=True)
    metrics_path = os.path.join("metrics", f"{algorithm}_{dataset}_{dirichlet}_{args.seed}_metrics.csv")
    metrics_results = {
        "epoch": np.arange(1, args.n_epochs+1),
        "train_losses": metrics["train_loss"],
        "train_accs": metrics["train_acc"],
        "val_losses": metrics["val_loss"],
        "val_accs": metrics["val_acc"]
    }

    pd.DataFrame(metrics_results).to_csv(metrics_path, index=False)

    print(f"\nMetrics saved to {metrics_path}")

    avg_comm_bytes = np.mean(metrics["comm_costs"])
    avg_train_times = np.mean(metrics["train_times"])
    
    # Log results
    results_path = "results.csv"
    final_results = {
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

    pd.DataFrame([final_results]).to_csv(results_path,
                                    mode='a',
                                    header=not os.path.exists(results_path),
                                    index=False)
    
    print(f"Results saved to {results_path}")

    # Plot curves
    os.makedirs("plots", exist_ok=True)
    path1 = plot_curve(algorithm, dirichlet, args.seed, "accuracy",
                   metrics["train_acc"], metrics["val_acc"], "plots")
    path2 = plot_curve(algorithm, dirichlet, args.seed, "loss",
                    metrics["train_loss"], metrics["val_loss"], "plots")

    print(f"Plots saved to {path1} and {path2}\n")


def get_data(dataset, n_clients, dir_alpha, batch_size):
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


def client_update(model, train_loader, epochs, device, algorithm, global_params=None, prox_mu=None):
    model = deepcopy(model).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)

            # Add FedProx proximal term
            if algorithm == "fedprox" and global_params is not None:
                prox_loss = 0.0
                for (name, param) in model.named_parameters():
                    prox_loss += ((param - global_params[name].to(param.device)) ** 2).sum()
                loss += (prox_mu / 2.0) * prox_loss

            loss.backward()
            optimizer.step()

    return model.state_dict()


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
    acc = round(100.0 * correct / total, 2)

    if split == "test":
        # Compute precision, recall, F1 (macro average = treats all classes equally)
        precision = round(100*precision_score(all_labels, all_preds, average="macro", zero_division=0), 2)
        recall = round(100*recall_score(all_labels, all_preds, average="macro", zero_division=0), 2)
        f1 = round(100*f1_score(all_labels, all_preds, average="macro", zero_division=0), 2)
        
        return acc, avg_loss, f1, precision, recall

    else:
        return acc, avg_loss


def federated_training(global_model, algorithm, train_loader, global_train_loader, 
                        val_loader, test_loader, n_clients, rounds, prox_mu, 
                        participation, local_epochs, sparsity, device, acg_momentum):

    global_model.to(device)
    pretrained_state = deepcopy(global_model.state_dict())

    # Initialise previous global state for FedACG 
    if algorithm == "fedacg":
        prev_global_state = deepcopy(global_model.state_dict())

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

        # FedACG lookahead
        if algorithm == "fedacg":
            global_model = FedACG_lookahead(
                model=global_model,
                prev_global_state=prev_global_state,
                acg_momentum=acg_momentum
            )

        idx_clients = np.random.choice(range(n_clients), m, replace=False)
        client_deltas_list, comm = [], 0

        print(f"\nTraining clients", end="")

        for client_idx in idx_clients:
            print(".", end="")
            
            # Point sampler to this client
            train_loader.sampler.set_client(int(client_idx))

            # Train client locally
            if algorithm == "fedprox":
                global_params = deepcopy(global_model.state_dict())
                client_state = client_update(
                    global_model, train_loader, local_epochs, device,
                    algorithm="fedprox", global_params=global_params, prox_mu=prox_mu
                )
            else:
                client_state = client_update(
                    global_model, train_loader, local_epochs, device,
                    algorithm=algorithm
                )

            if algorithm == "las":
                # Localise deltas
                localiser = Localiser(pretrained_state, sparsity)
                _, deltas, calc_comm = localiser.compute_mask_and_deltas(client_state)
                comm += calc_comm
                client_deltas_list.append(deltas)
            else:
                # Collect full client states
                client_deltas_list.append(client_state)
        
        # Aggregation step
        if algorithm == "las":
            # Stitch results into new global state
            stitcher = Stitcher(pretrained_state)
            new_global_state = stitcher.stitch(client_deltas_list)
        elif algorithm in ["fedavg", "fedprox"]:
            # Average weights
            new_global_state, comm = average_weights(client_deltas_list)
        elif algorithm == "fedacg":
            new_global_state, comm = FedACG_aggregate(global_model, client_deltas_list)
            prev_global_state = deepcopy(new_global_state)
        else:
            raise ValueError(f"Invalid algorithm name, {algorithm}")

        metrics["comm_costs"].append(comm)

        # Update global model
        global_model.load_state_dict(new_global_state)
        pretrained_state = deepcopy(new_global_state)

        # Evaluate using train and validation sets
        train_acc, train_loss = evaluate(global_model, global_train_loader, device, split="train")
        val_acc, val_loss = evaluate(global_model, val_loader, device, split="val")

        # Print results
        print(f"\nResults after {r + 1} rounds of training:")
        print(f"---> Avg Train Loss: {train_loss} | Avg Train Accuracy: {train_acc}%")
        print(f"---> Avg Val Loss: {val_loss} | Avg Val Accuracy: {val_acc}%")
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
    print(f"---> Accuracy: {test_acc}%")
    print(f"---> F1 Score: {f1}%")
    print(f"---> Precision: {precision}%")
    print(f"---> Recall: {recall}%")

    return global_model, metrics, test_acc, f1, precision, recall


if __name__ == "__main__":
    main()