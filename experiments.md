# Non-IID Experiments with CIFAR10

## Fedavg

### MLP
python main.py 
--algorithm fedavg 
--model_name mlp 
--n_clients 10 
--frac 0.5 
--n_epochs 100 
--n_client_epochs 5 
--batch_size 32 
--lr 0.01 
--momentum 0.9 

### CNN
python main.py 
--algorithm fedavg 
--model_name cnn 
--n_clients 10 
--frac 0.5 
--n_epochs 100 
--n_client_epochs 5 
--batch_size 32 
--lr 0.01 
--momentum 0.9 

## LaS

### MLP
python main.py 
--algorithm las 
--model_name mlp 
--n_clients 10 
--frac 0.5 
--n_epochs 100 
--n_client_epochs 5 
--batch_size 32 
--lr 0.01 
--momentum 0.9 
--graft_epochs 10 
--topk 0.2 
--sparsity 0.2 
--l1_strength 0.0 

### CNN
python main.py 
--algorithm las  
--dataset CIFAR10 
--model_name cnn 
--n_clients 10 
--frac 0.5 
--n_epochs 100 
--n_client_epochs 5 
--batch_size 32 
--lr 0.01 
--momentum 0.9 
--graft_epochs 10 
--topk 0.2 
--sparsity 0.2 
--l1_strength 0.0 


## original setup
python main.py 
--algorithm fedavg 
--model_name cnn 
--n_clients 10 
--frac 0.5 
--n_epochs 50 
--n_client_epochs 5 
--batch_size 32 
--lr 0.001 
--momentum 0.9

# Results after 50 rounds of CPU training:
Avg Val Accuracy: 29.10%
Training took 3022.6 seconds (avg 60s per round)
Avg Test Accuracy: 29.68%


# Results after 50 rounds of HPC training: (1214427)
Avg Val Accuracy: 23.58%
Training took 825.7 seconds (avg 17s per round)
Avg Test Accuracy: 23.07%


## new setup
python main.py 
--algorithm las 
--model_name cnn 
--n_clients 10 
--frac 0.5 
--n_epochs 100 
--n_client_epochs 5 
--batch_size 32 
--lr 0.001 
--momentum 0.9

# Results after 100 rounds of HPC training: (1214475)
Avg Val Accuracy: 9.96%
Training took 3568.5 seconds (avg 17s per round)
Avg Test Accuracy: 9.70%