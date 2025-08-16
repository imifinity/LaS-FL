from utils import arg_parser
from fedalg import FedAlg


def main():
    # Get args
    args = arg_parser()
    print(args.algorithm + "\n" + args.dataset)

    # Get federated algorithm
    fed_alg = FedAlg(args)

    # Train and test the algorithm
    fed_alg.train()
    fed_alg.test()


if __name__ == "__main__":
    main()