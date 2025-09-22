import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, random_split
from torchvision import datasets, transforms
import os
import random


class CIFAR10Dataset(datasets.CIFAR10): # CIFAR10
    N_CLASSES = 10
    def __init__(self, root="./data", train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2023, 0.1994, 0.2010)),
        ])
        super().__init__(root=root, train=train, download=False, transform=transform)

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]
        x = self.transform(x)
        return x, y

class CIFAR10TinyDataset(datasets.CIFAR10): # CIFAR10T
    N_CLASSES = 10
    def __init__(self, root="./data", train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        super().__init__(root=root, train=train, download=False, transform=transform)

        indices = random.sample(range(len(self.data)), 1000)
        self.data = self.data[indices]
        self.targets = [self.targets[i] for i in indices]

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]
        x = self.transform(x)
        return x, y

class TinyImageNetCachedDataset(Dataset): # TinyImageNet
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
        return len(self.targets)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


def load_datasets(dataset="CIFAR10"):
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
    def __init__(self, dataset, n_clients, dir_alpha):
        """Sampler for federated learning in both iid and non-iid settings.

        Args:
            dataset (Sequence): Dataset to sample from.
            non_iid (int): 0: IID, 1: Non-IID
            n_clients (Optional[int], optional): Number of clients.
            dir_alpha (Optional[int], optional): Dirichlet dist. param.
        """
        self.dataset = dataset
        self.n_clients = n_clients
        self.dir_alpha = dir_alpha

        if np.isnan(self.dir_alpha):
            self.dict_users = self._sample_iid()
        else:
            self.dict_users = self._sample_non_iid()

    def _sample_iid(self):
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
        self.client_id = client_id

    def __iter__(self):
        # fetch dataset indexes based on current client
        client_idxs = list(self.dict_users[self.client_id])
        for item in client_idxs:
            yield int(item)
    
    def __len__(self):
        if hasattr(self, 'client_id'):
            return len(self.dict_users[self.client_id])
        else:
            return len(self.dataset)