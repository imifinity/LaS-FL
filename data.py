import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, random_split
from torchvision import datasets, transforms
import os
import random


class CIFAR10Dataset(datasets.CIFAR10): # CIFAR10
    """ CIFAR-10 dataset wrapper with standard normalisation.""" 
    N_CLASSES = 10

    def __init__(self, root="./data", train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2023, 0.1994, 0.2010)),
        ])
        super().__init__(root=root, train=train, download=False, transform=transform)

    def __getitem__(self, index):
        """Return transformed image and label at given index."""
        x, y = self.data[index], self.targets[index]
        x = self.transform(x)
        return x, y


class CIFAR10TinyDataset(datasets.CIFAR10): # CIFAR10T
    """ Small random subset of CIFAR-10 (1000 samples)."""
    N_CLASSES = 10

    def __init__(self, root="./data", train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        super().__init__(root=root, train=train, download=False, transform=transform)

        # Randomly sample 1000 examples
        indices = random.sample(range(len(self.data)), 1000)
        self.data = self.data[indices]
        self.targets = [self.targets[i] for i in indices]

    def __getitem__(self, index):
        """Return transformed image and label at given index."""
        x, y = self.data[index], self.targets[index]
        x = self.transform(x)
        return x, y


class TinyImageNetCachedDataset(Dataset): # TinyImageNet
    """ TinyImageNet dataset loaded from pre-computed cache."""

    def __init__(self, data_dir="./data", split="train"):
        cache_file = os.path.join(data_dir, f"tinyimagenet_{split}_cache.pt")
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file not found: {cache_file}. "
                                    "Run build_tinyimagenet_cache.py first.")
        print(f"Loading TinyImageNet {split} from cache...")
        cache = torch.load(cache_file)
        self.samples = cache["samples"]
        self.targets = cache["targets"]

    def __len__(self):
        """Return number of samples."""
        return len(self.targets)

    def __getitem__(self, idx):
        """Return sample and label at given index."""
        return self.samples[idx], self.targets[idx]


def load_datasets(dataset="CIFAR10"):
    """
    Load dataset splits for training, validation, and testing.

    Args:
        dataset (str): One of ["CIFAR10", "TinyImageNet", "CIFAR10T"]

    Returns:
        train_ds (Dataset): Training set
        val_ds (Dataset): Validation set (10% of training data)
        test_ds (Dataset): Test set
    """

    if dataset == "CIFAR10":
        train_ds = CIFAR10Dataset(train=True)
        test_ds    = CIFAR10Dataset(train=False)
        
    elif dataset == "TinyImageNet":
        train_ds = TinyImageNetCachedDataset(split="train")
        test_ds    = TinyImageNetCachedDataset(split="val")

    elif dataset == "CIFAR10T":
        train_ds = CIFAR10TinyDataset(train=True)
        test_ds    = CIFAR10TinyDataset(train=False)
        
    else:
        raise ValueError(f"Invalid dataset name, {dataset}")

    # Split training into train/val (90%/10%)
    n = len(train_ds)
    n_val = int(0.1 * n)
    n_train = n - n_val
    
    train_ds, val_ds = random_split(
        train_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    return train_ds, val_ds, test_ds


class FederatedSampler(Sampler):
    """
    Sampler for partitioning datasets across federated clients.

    Args:
        dataset (Dataset): Dataset to partition
        n_clients (int): Number of clients
        dir_alpha (float): Dirichlet parameter. If NaN, defaults to IID.
    """
    def __init__(self, dataset, n_clients, dir_alpha):
        self.dataset = dataset
        self.n_clients = n_clients
        self.dir_alpha = dir_alpha

        # Choose sampling strategy
        if np.isnan(self.dir_alpha):
            self.dict_users = self._sample_iid()
        else:
            self.dict_users = self._sample_non_iid()

    def _sample_iid(self):
        """Partition dataset equally at random among clients."""
        num_items = len(self.dataset) // self.n_clients
        dict_users, all_idxs = {}, [i for i in range(len(self.dataset))]

        for i in range(self.n_clients):
            if i == self.n_clients - 1:  # last client takes the remainder
                dict_users[i] = set(all_idxs)
            else:
                dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
                all_idxs = list(set(all_idxs) - dict_users[i])

        return dict_users

    def _sample_non_iid(self):
        """Partition dataset using Dirichlet distribution over classes."""
        # Extract labels
        if hasattr(self.dataset, 'dataset') and hasattr(self.dataset.dataset, 'targets'):
            all_indices = np.arange(len(self.dataset))
            labels = np.array([self.dataset.dataset.targets[i] for i in all_indices])
        else:
            labels = np.array(self.dataset.targets)
            all_indices = np.arange(len(self.dataset))

        num_classes = len(np.unique(labels))
        dict_users = {i: [] for i in range(self.n_clients)}

        # Group indices by class
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

        # Allocate samples for each class based on Dirichlet proportions
        for c, indices in enumerate(class_indices):
            np.random.shuffle(indices)
            proportions = np.random.dirichlet([self.dir_alpha] * self.n_clients)
            proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            splits = np.split(indices, proportions)

            for i, client_indices in enumerate(splits):
                dict_users[i].extend(client_indices)

        # Convert lists to numpy arrays
        for i in dict_users:
            dict_users[i] = np.array(dict_users[i])

        return dict_users

    def set_client(self, client_id: int):
        """Fix current client for sampling."""
        self.client_id = client_id

    def __iter__(self):
        """Get indices for current client's partition."""
        client_idxs = list(self.dict_users[self.client_id])
        for item in client_idxs:
            yield int(item)
    
    def __len__(self):
        """Return size of current client's partition, or full dataset if unset."""
        if hasattr(self, 'client_id'):
            return len(self.dict_users[self.client_id])
        else:
            return len(self.dataset)