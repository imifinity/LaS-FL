from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import os

from utils import arg_parser
# from fedalg import FedAlg
from fedlas import FedAlg

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_curve(metric, train, val, filepath, algorithm):

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
    path = os.path.join(filepath, f"{algorithm}_{metric}_curves_{datetime.now().strftime('%d-%m_%H-%M-%S')}.png")
    plt.savefig(path)


def main():
    # Set the seed for reproducibility
    set_seed(42)

    # Get args
    args = arg_parser()
    print(args.algorithm + "\n" + args.dataset)
    print(args.dirichlet)

    # Get federated algorithm
    fed_alg = FedAlg(args)

    # Train and test the algorithm
    train_losses, val_losses, train_accs, val_accs, filename = fed_alg.train()
    fed_alg.evaluate(split="test")

    # Plot training and validation curves
    plot_curve("accuracy", train_accs, val_accs, filename, args.algorithm)
    plot_curve("loss", train_losses, val_losses, filename, args.algorithm)

if __name__ == "__main__":
    main()