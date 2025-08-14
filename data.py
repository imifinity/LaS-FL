from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, random_split, ConcatDataset
from torchvision import datasets, transforms
import pandas as pd
from PIL import Image
from io import BytesIO


class CIFAR10Dataset(datasets.CIFAR10):
    N_CLASSES = 10
    def __init__(self, root: str, train: bool):
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


class TinyImageNetDataset:
    N_CLASSES = 200
    def __init__(self, split='train'):
        splits = {'train': 'data/train.parquet', 'valid': 'data/valid.parquet'}
        self.ds = pd.read_parquet(splits[split])
        self.targets = self.ds["label"].tolist()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2302, 0.2265, 0.2262)),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img_dict = self.ds.iloc[index]["image"]
        img_bytes = img_dict['bytes']
        img = Image.open(BytesIO(img_bytes)).convert("RGB")  # decode bytes to RGB PIL image

        x = self.transform(img)
        y = self.targets[index]
        return x, y


def load_datasets(dataset="CIFAR10"):
    if dataset == "CIFAR10":
        train_ds = CIFAR10Dataset(root="./data", train=True)
        test_ds    = CIFAR10Dataset(root="./data", train=False)
        
    elif dataset == "TinyImageNet":
        train_ds = TinyImageNetDataset(split="train")
        test_ds    = TinyImageNetDataset(split="valid")

    else:
        raise ValueError(f"Invalid dataset name, {self.args.dataset}")

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
    def __init__(
        self,
        dataset: Sequence,
        non_iid: int,
        n_clients: Optional[int] = 100,
        dir_alpha: Optional[float] = 0.5,
    ):
        """Sampler for federated learning in both iid and non-iid settings.

        Args:
            dataset (Sequence): Dataset to sample from.
            non_iid (int): 0: IID, 1: Non-IID
            n_clients (Optional[int], optional): Number of clients. Defaults to 100.
            dir_alpha (Optional[int], optional): Dirichlet dist. param. Defaults to 0.5.
        """
        self.dataset = dataset
        self.non_iid = non_iid
        self.n_clients = n_clients
        self.dir_alpha = dir_alpha

        if self.non_iid:
            self.dict_users = self._sample_non_iid()
        else:
            self.dict_users = self._sample_iid()

    def _sample_iid(self) -> Dict[int, List[int]]:
        num_items = len(self.dataset) // self.n_clients
        dict_users, all_idxs = {}, [i for i in range(len(self.dataset))]

        for i in range(self.n_clients):
            if i == self.n_clients - 1:  # last client takes the remainder
                dict_users[i] = set(all_idxs)
            else:
                dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
                all_idxs = list(set(all_idxs) - dict_users[i])

        return dict_users

    def _sample_non_iid(self) -> Dict[int, List[int]]:
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