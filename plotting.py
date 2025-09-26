import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
from datetime import datetime

def arg_parser():
    """Collect command-line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_dir", type=str, default="results",
                        help="Directory containing CSV files with logged metrics")
    parser.add_argument("--output_dir", type=str, default="plots",
                        help="Directory to save generated plots")
    parser.add_argument("--algorithm", type=str, default="las",
                        help="Federated learning aggregation method to filter plots")
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        help="Dataset to filter plots")

    return parser.parse_args()


def plot_las_curves(dset, algorithm, csv_dir, output_dir):
    """
    Plot training and validation accuracy curves for the LaS algorithm
    across different Dirichlet splits (IID, 0.1, 1.0).
    
    Args:
        dset (str): Dataset name to filter CSV files
        algorithm (str): Algorithm name (used in output filename)
        csv_dir (str): Directory containing CSV logs
        output_dir (str): Directory to save the plots
    """

    plt.figure(figsize=(8, 5))

    metric = "accs"
    col_train = f"train_{metric}"
    col_val = f"val_{metric}"

    # Loop through all CSV files
    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            filename = os.path.basename(file).split("_")  # e.g. "fedacg_CIFAR10_0.1_42_metrics.csv"
            algo = filename[0]
            dataset = filename[1]
            diri = filename[2]
            filepath = os.path.join(csv_dir, file)

            # Only consider chosen dataset and LaS algorithm
            if dataset != dset or algo != "las":
                continue

            df = pd.read_csv(filepath)

            # Skip if metrics are missing
            if col_train not in df.columns or col_val not in df.columns:
                print(f"Skipping {file}, missing {col_train} or {col_val}")
                continue

            # Map Dirichlet splits to consistent colours
            algo_colors = {
                "IID": "tab:blue",
                "0.1": "tab:orange",
                "1.0": "tab:green"
            }

            epochs = range(1, len(df[col_train]) + 1)
            if diri == "nan": diri = "IID"
            color = algo_colors.get(diri, "black")

            # Plot train (solid) + val (dashed) for each Dirichlet
            plt.plot(epochs, df[col_train], color=color, label=f"Dir {diri} (train)", linestyle="-")
            plt.plot(epochs, df[col_val], color=color, label=f"Dir {diri} (val)", linestyle="--")

    plt.title(f"LaS Training and Validation Accuracy Curves on {dataset}")
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # Save plot with timestamp for uniqueness
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{dataset}_{algorithm}_{metric}_curves_{datetime.now().strftime('%m-%d__%H-%M-%S')}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {path}")


def plot_val_curves(dset, csv_dir, output_dir):
    """
    Plot validation accuracy curves for all algorithms on a given dataset.
    
    Args:
        dset (str): Dataset name to filter CSV files
        csv_dir (str): Directory containing CSV logs
        output_dir (str): Directory to save the plots
    """

    plt.figure(figsize=(8, 5))
    col = "val_accs"

    # Loop through all CSV files in directory
    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            filename = os.path.basename(file).split("_")  # e.g. "fedacg_CIFAR10_0.1_42_metrics.csv"
            algorithm = filename[0]
            dataset = filename[1]
            diri = filename[2]
            filepath = os.path.join(csv_dir, file)

            # Only consider chosen dataset
            if dataset != dset:
                continue

            # Load CSV
            df = pd.read_csv(filepath)

            if col not in df.columns:
                print(f"Skipping {file}, no '{col}' column found")
                continue

            epochs = range(1, len(df[col]) + 1)
            
            # Assign colours based on algorithm
            if algorithm == "fedavg": color="tab:blue"
            elif algorithm == "fedacg": color="tab:red"
            elif algorithm == "fedprox": color="tab:orange"
            else: color="tab:green"

            # Assign linestyles based on Dirichlet split
            if diri == "0.1": linestyle="-"
            elif diri == "1.0": linestyle="--"
            else: 
                linestyle=":"
                diri = "IID"

            plt.plot(epochs, df[col], color=color, linestyle=linestyle, linewidth=1.1, label=f"{algorithm}_{diri}")

    plt.title(f"Validation Accuracy Curves on {dataset}")
    plt.xlabel("Epoch")
    plt.ylabel(f"Accuracy (%)")

    # Sort legend entries for readability
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda x: x[0]))
    plt.legend(handles, labels)
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{dataset}_{col}_curves_{datetime.now().strftime('%m-%d__%H-%M-%S')}.png")
    plt.savefig(path)
    print(f"Plot saved to {path}")


def plot_dir_curves(dset, dirichlet, csv_dir, output_dir):
    """
    Plot training and validation accuracy curves for all algorithms
    on a fixed dataset and fixed Dirichlet split.
    
    Args:
        dset (str): Dataset name to filter CSV files
        dirichlet (str): Dirichlet alpha value (e.g. "nan", "0.1", "1.0")
        csv_dir (str): Directory containing CSV logs
        output_dir (str): Directory to save the plots
    """

    plt.figure(figsize=(8, 5))

    col_train = "train_accs"
    col_val = "val_accs"

    # Loop through all CSV files in directory
    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            filename = os.path.basename(file).split("_")  # e.g. "fedacg_CIFAR10_0.1_42_metrics.csv"
            algorithm = filename[0]
            dataset = filename[1]
            diri = filename[2]
            filepath = os.path.join(csv_dir, file)

            # Only consider chosen dataset and Dirichlet split
            if dataset != dset or diri != dirichlet:
                continue

            df = pd.read_csv(filepath)

            if col_train not in df.columns or col_val not in df.columns:
                print(f"Skipping {file}, missing {col_train} or {col_val}")
                continue

            # Assign colours consistently to algorithms
            algo_colors = {
                "fedavg": "tab:blue",
                "fedacg": "tab:red",
                "fedprox": "tab:orange",
                "las": "tab:green"
            }

            epochs = range(1, len(df[col_train]) + 1)
            color = algo_colors.get(algorithm, "black")

            # Training curve
            plt.plot(epochs, df[col_train],
                     label=f"{algorithm} (train)",
                     color=color,
                     linestyle="-",
                     linewidth=1.0)

            # Validation curve
            plt.plot(epochs, df[col_val],
                     label=f"{algorithm} (val)",
                     color=color,
                     linestyle="--",
                     linewidth=1.0)

    if dirichlet == "nan": dirichlet = "IID"

    plt.title(f"Training and Validation Accuracy Curves on {dataset} (Dirichlet={dirichlet})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(
        output_dir,
        f"{dset}_{dirichlet}_accs_curves_{datetime.now().strftime('%m-%d__%H-%M-%S')}.png"
    )
    plt.savefig(path)
    plt.close() # close figure to free memory when looping 
    print(f"Plot saved to {path}")


def main():
    """Main entry point: parse args and generate plots"""
    args = arg_parser()
    plot_las_curves(args.dataset, args.algorithm, args.csv_dir, args.output_dir)
    plot_val_curves(args.dataset, args.csv_dir, args.output_dir)
    plot_dir_curves(args.dataset, "nan", args.csv_dir, args.output_dir)
    plot_dir_curves(args.dataset, "1.0", args.csv_dir, args.output_dir)
    plot_dir_curves(args.dataset, "0.1", args.csv_dir, args.output_dir)


if __name__ == "__main__":
    main()