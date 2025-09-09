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


def plot_train_curve(losses, filepath, algorithm):

    # Plot training loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title(f"{algorithm} Training Loss Curve")
    plt.xlabel("Round (Epoch)")
    plt.ylabel("Average Client Training Loss")
    plt.grid(True)
    
    # Save plot as PNG file
    path = os.path.join(filepath, f"training_loss_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
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
    train_losses, filename = fed_alg.train()
    plot_train_curve(train_losses, filename, args.algorithm)
    fed_alg.test()


if __name__ == "__main__":
    main()