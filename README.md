# Localised Model Merging for Federated Learning

This repository contains the implementation of several federated learning algorithms, including **FedAvg**, **FedProx**, **FedAcg**, and our implementation, **LaS-FL**, evaluated on datasets CIFAR-10 and TinyImageNet.  

The project addresses the following research question:
“Can localised model merging methods like ‘Localize-and-Stitch’ be effectively adapted to federated learning environments in a way that maintains high performance while reducing communication and storage costs across different levels of data heterogeneity?”

## Abstract:
Federated Learning enables collaborative training without centralising raw data, but remains limited by communication bottlenecks, data heterogeneity, and slower convergence compared to centralised methods. This dissertation explores Localize-and-Stitch for Federated Learning (LaS-FL), an adaptation of a localised model merging approach proposed by He et al. (2025), as an alternative to standard, global aggregation techniques. Rather than averaging full model updates, LaS-FL selectively merges subsets of parameters, aiming to reduce communication while maintaining accuracy.

LaS-FL was evaluated on CIFAR-10 and Tiny-ImageNet under varying data distributions, with comparisons based on accuracy, F1-score, communication cost, and average training time. On CIFAR-10, LaS-FL achieved comparable or stronger performance than baselines, while on Tiny-ImageNet it lagged by around 5% but remained competitive. Crucially, across both datasets LaS-FL reduced communication by a factor of five to six per round. 

These findings suggest that localised model merging can be feasibly integrated into FL, offering meaningful efficiency gains. At the same time, results highlight the need for refinement to ensure robustness on larger, more heterogeneous datasets. This work therefore represents an initial step towards communication-efficient federated training with localised model merging.

---

# Requirements

Setup environment then install dependencies with:
pip install -r requirements.txt

# Project Structure
```
.
├── data/               # Empty - all data for CIFAR10 and TinyImageNet to be downloaded here
├── final_plots/        # Final training/validation accuracy plots for comparison
├── metrics/            # Raw training/validation performance metrics for each experiment
├── plots/              # Accuracy/loss line plots for each experiment
├── README.md           # Project documentation (this file)
├── agg_utils.py        # Aggregation algorithms: FedAvg, FedProx, FedACG, LaS-FL
├── data.py             # Data loading and preprocessing
├── experiments.txt     # Parameter combinations for experiments (used by run_experiments.sh)
├── fedalg.py           # Federated training/testing loop (includes main())
├── import_TIN.py       # Imports Tiny-ImageNet and caches it in ./data/
├── models.py           # ResNet18/ResNet50 architectures
├── plotting.py         # Generates plots from experiment results
├── plotting.sh         # SLURM script for plotting.py
├── requirements.txt    # Python dependencies
├── results.csv         # Raw test results
├── run_experiments.sh  # SLURM script for fedalg.py
└── utils.py            # Argument parser, plotting utilities, seed initialiser
```

# Datasets
## CIFAR-10
Manually download CIFAR-10 python version:
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O ./data/cifar10/cifar-10-python.tar.gz
tar -xvzf ./data/cifar-10-python.tar.gz -C ./data/

## TinyImageNet
Manually download TinyImageNet:
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d ./data/
then run: import_tin.py to cache the dataset

The final structure should look like this:
```
./data/
├── cifar-10-batches-py/
│   ├── batches.meta
│   ├── cifar10.tar.gz
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── readme.html
│   └── test_batch
├── tinyimagenet_train_cache.pt
└── tinyimagenet_val_cache.pt
```

# Running the Code
All command-line args can be found in ./utils.py

##  Local Execution
Example (FedAvg on CIFAR-10):
python main.py --algorithm fedavg --dataset CIFAR10 --dirichlet 0.1 --n_epochs 100 --seed 42

## HPC (SLURM) Execution
If running on a SLURM-based HPC, use the provided job script: run_experiments.sh

Example submission:
sbatch scripts/run_experiment.sh

# Plotting graphs
Loss and accuracy graphs are automatically plotted and stored in ./plots/ after running fedalg.py
There is the option to produce additional graphs that are stored in ./final plots/ by running the following for example:

python plotting.py --csv_dir metrics --output_dir final_plots --dataset CIFAR10

# Notes

.sh job scripts are included for reproducibility on HPC.

On local machines, you can ignore them and just run the python files directly

Reference: He, Y., Hu, Y., Lin, Y., Zhang, T. & Zhao, H. (2025), ‘Localize-and-stitch: Efficient model merging via sparse task arithmetic’. Available at: https://arxiv.org/abs/2408.13656
