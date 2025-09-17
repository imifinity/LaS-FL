from utils import arg_parser
from fedalg import FedAlg

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import os


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_curve(mode, accs, losses, filepath, algorithm):

    # Plot training loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(accs) + 1), accs, marker='o')
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title(f"{algorithm.title()} {mode.title()} Curves")
    plt.xlabel("Round (Epoch)")
    plt.ylabel(f"{mode.title()} metrics")
    plt.grid(True)
    plt.legend()
    
    # Save plot as PNG file
    path = os.path.join(filepath, f"{algorithm}_{mode}_curves_{datetime.now().strftime('%S-%M-%H_%d-%m')}.png")
    plt.savefig(path)


def main():
    # Set the seed for reproducibility
    set_seed(42)

    # Get args
    args = arg_parser()
    print(args.algorithm + "\n" + args.dataset)
    print(args.device)

    # Get federated algorithm
    fed_alg = FedAlg(args)

    # Train and test the algorithm
    train_losses, val_losses, train_accs, val_accs, filename = fed_alg.train()
    fed_alg.test()

    # Plot training and validation curves
    plot_curve("train", train_accs, train_losses, filename, args.algorithm)
    plot_curve("val", val_accs, val_losses, filename, args.algorithm)

if __name__ == "__main__":
    main()