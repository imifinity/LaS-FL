import os
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures deterministic behaviour
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def arg_parser() -> argparse.ArgumentParser:
    """Collect command-line arguments"""
    parser = argparse.ArgumentParser()

    # Args for general setup
    parser.add_argument("--data_root", type=str, default="./data/",
                        help="Path to the dataset directory")
    parser.add_argument("--algorithm", type=str, default="fedavg",
                        help="Federated learning aggregation method to use ")
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        help="Dataset to train on")
    parser.add_argument("--model_name", type=str, default="resnet50",
                        help="Model architecture")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for reproducibility")

    # Federation args
    parser.add_argument("--dirichlet", type=float, default=np.nan,
                        help="Dirichlet alpha parameter for non-IID data split (np.nan = IID)")
    parser.add_argument("--n_clients", type=int, default=20,
                        help="Total number of federated clients")
    parser.add_argument("--participation", type=float, default=0.2,
                        help="Fraction of clients participating per round")

    # Training args
    parser.add_argument("--n_epochs", type=int, default=100,
                        help="Number of global training rounds")
    parser.add_argument("--n_client_epochs", type=int, default=5,
                        help="Number of local epochs per client")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for client training")

    # Aggregation args 
    parser.add_argument('--sparsity', type=float, default=0.2,
                        help="Proportion of parameters retained in sparse aggregation for LaS")
    parser.add_argument("--prox_momentum", type=float, default=0.01,
                        help="Proximal momentum term for FedProx")
    parser.add_argument("--acg_momentum", type=float, default=0.1,
                        help="Adaptive control gradient momentum for FedACG")

    return parser.parse_args()


def plot_curve(algorithm, dirichlet, seed, metric, train, val, filepath):
    """Plot training and validation curves and save to file"""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train) + 1), train, marker='o', label='train')
    plt.plot(range(1, len(val) + 1), val, marker='o', label='validation')
    plt.title(f"{metric.title()} Curves")
    plt.xlabel("Round (Epoch)")
    plt.ylabel(metric.title())
    plt.grid(True)
    plt.legend()
    
    # Save plot as PNG file under unique filename
    path = os.path.join(filepath, f"{algorithm}_{dirichlet}_{seed}_{metric}_curves.png")
    plt.savefig(path)

    return path