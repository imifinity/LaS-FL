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

    # Aggregation args 
    parser.add_argument('--sparsity', type=float, default=0.2)
    parser.add_argument("--prox_momentum", type=float, default=0.01)
    parser.add_argument("--acg_momentum", type=float, default=0.1)

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