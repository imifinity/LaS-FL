from utils import arg_parser
from fedalg import FedAlg
import numpy as np
import random
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    fed_alg.train()
    fed_alg.test()


if __name__ == "__main__":
    main()