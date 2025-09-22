import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from datetime import datetime

def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--dirichlet", type=str, default="plots")
    parser.add_argument("--dataset", type=str, default="plots",)

    return parser.parse_args()


def plot_curves(dirichlet, dataset, mode, metric, csv_dir, output_dir):
    plt.figure(figsize=(8, 5))
    
    if mode == "val":
        col = f"val_{metric}"
        name = "Validation"
    elif mode == "train":
        col = f"train_{metric}"
        name = "Training"
    else:
        print(f"{mode} is not a valid mode")
    # Loop through all CSV files in directory
    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            algorithm = file.replace("_metrics.csv", "")  # assume filename is algorithm_metrics.csv
            filepath = os.path.join(csv_dir, file)

            # Load CSV
            df = pd.read_csv(filepath)

            if col not in df.columns:
                print(f"Skipping {file}, no '{col}' column found")
                continue

            epochs = range(1, len(df[col]) + 1)
            plt.plot(epochs, df[col], label=algorithm)

    if metric == "accs": metric2 = "accuracy" 
    else: metric2 = "loss"

    plt.title(f"{name} {metric2.title()} Curves")
    plt.xlabel("Epoch")
    plt.ylabel(f"{name} {metric2.title()}")
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{dirichlet}_{dataset}_{col}_curves_{datetime.now().strftime('%m-%d__%H-%M-%S')}.png")
    plt.savefig(path)
    print(f"Plot saved to {path}")


def main():
    args = arg_parser()
    plot_curves(args.dataset, args.dirichlet, "val", "losses", args.csv_dir, args.output_dir)
    plot_curves(args.dataset, args.dirichlet, "val", "accs", args.csv_dir, args.output_dir)
    plot_curves(args.dataset, args.dirichlet, "train", "losses", args.csv_dir, args.output_dir)
    plot_curves(args.dataset, args.dirichlet, "train", "accs", args.csv_dir, args.output_dir)

if __name__ == "__main__":
    main()
