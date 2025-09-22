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
├── main.py             # Entry point, parses args & starts training and evaluates result
├── fedalg.py           # Federated learning training loop with a test function included
├── utils.py            # Argument parser, and aggregation algorithms for: FedAvg, FedProx, FedACG, and LaS
├── models.py           # ResNet18/ResNet50 architectures
├── data.py             # Data loading & preprocessing
├── requirements.txt    # Dependencies
├── scripts/            # SLURM job scripts
├── data/               # Empty - all data for CIFAR10 and TinyImageNet to be downloaded here
├── extra/
│ └── import_TIN.py
└── README.md
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
then run: python import_tin.py to cache the dataset

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
All args can be found in ./utils.py

##  Local Execution
Example (FedAvg on CIFAR-10):
python main.py --algorithm fedavg --dataset CIFAR10 --dirichlet 0.1 --n_epochs 50 --seed 42

## HPC (SLURM) Execution
If running on a SLURM-based HPC, use the provided job scripts in ./scripts/.

Example submission:
sbatch scripts/run_experiment.sh

# Notes

.sh job scripts are included in ./scripts/ for reproducibility on HPC.

On local machines, you can ignore them and just run python main.py ....

Results are written to the output directory specified in your job script.
