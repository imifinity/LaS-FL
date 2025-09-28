# Localised Model Merging for Federated Learning

This repository contains the implementation of several federated learning algorithms, including **FedAvg**, **FedProx**, **FedAcg**, and **Localize-and-Stitch (LaS)**, evaluated on datasets CIFAR-10 and TinyImageNet.  

The project addresses the following research question:
“Can localised model merging methods like ‘Localize-and-Stitch’ be effectively adapted to federated learning environments in a way that maintains high performance while reducing communication and storage costs across different levels of data heterogeneity?”

Abstract:
...

---

# Requirements

Setup environment then install dependencies with:
pip install -r requirements.txt

# Project Structure
```
.
├── data/               # Empty - all data for CIFAR10 and TinyImageNet to be downloaded here
├── README.md           # This file
├── agg_utils.py        # Aggregation algorithms for FedAvg, FedProx, FedACG, and LaS-FL
├── data.py             # Data loading & preprocessing
├── experiments.txt     # Parameter combinations for all experiments - used by run_experiments.sh
├── fedalg.py           # Contains multiple functions that train and test the federated learning setup - includes main()
├── import_TIN.py       # Imports Tiny-ImageNet and saves a cached version of it in ./data/
├── models.py           # ResNet18/ResNet50 architectures
├── plotting.py         # Plots graphs after experiments have run
├── plotting.sh         # SLURM script to run plotting.py
├── requirements.txt    # Dependencies
├── run_experiments.sh  # SLURM script to run fedalg.py
└── utils.py            # Argument parser, graph plotter, and seed initialiser
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
