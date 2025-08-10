from typing import Dict, List, Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, random_split
from torchvision import datasets, transforms
from datasets import load_dataset


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
        self.ds = load_dataset("zh-plus/tiny-imagenet", split=split, cache_dir="/users/adgs945/.cache/huggingface")
        self.targets = [item['label'] for item in self.ds]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                 (0.2302, 0.2265, 0.2262)),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        x, y = self.ds[index]['image'], self.ds[index]['label']
        if x.mode != "RGB":  # ensure RGB format
            x = x.convert("RGB")
        x = self.transform(x)
        return x, y


def load_datasets(dataset="CIFAR10"):
    if dataset == "CIFAR10":
        full_train = CIFAR10Dataset(root="./data", train=True)
        test_ds    = CIFAR10Dataset(root="./data", train=False)
        
    elif dataset == "TinyImageNet":
        full_train = TinyImageNetDataset(split="train")
        test_ds    = TinyImageNetDataset(split="valid")

    else:
        raise ValueError(f"Invalid dataset name, {self.args.dataset}")

    n_train    = len(full_train)
    n_val      = int(0.1 * n_train)
    n_train    = n_train - n_val
    
    train_ds, val_ds = random_split(
        full_train,
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
        n_shards: Optional[int] = 200,
    ):
        """Sampler for federated learning in both iid and non-iid settings.

        Args:
            dataset (Sequence): Dataset to sample from.
            non_iid (int): 0: IID, 1: Non-IID
            n_clients (Optional[int], optional): Number of clients. Defaults to 100.
            n_shards (Optional[int], optional): Number of shards. Defaults to 200.
        """
        self.dataset = dataset
        self.non_iid = non_iid
        self.n_clients = n_clients
        self.n_shards = n_shards

        if self.non_iid:
            self.dict_users = self._sample_non_iid()
        else:
            self.dict_users = self._sample_iid()

    def _sample_iid(self) -> Dict[int, List[int]]:
        num_items = len(self.dataset) // self.n_clients
        dict_users, all_idxs = {}, [i for i in range(len(self.dataset))]

        for i in range(self.n_clients):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])

        return dict_users

    def _sample_non_iid(self) -> Dict[int, List[int]]:
        num_imgs = len(self.dataset) // self.n_shards

        idx_shard = [i for i in range(self.n_shards)]
        dict_users = {i: np.array([]) for i in range(self.n_clients)}
        idxs = np.arange(self.n_shards * num_imgs)
        
        if hasattr(self.dataset, 'dataset') and hasattr(self.dataset.dataset, 'targets'):
            # For Subset objects
            labels = np.array([self.dataset.dataset.targets[i] for i in self.dataset.indices])
        else:
            labels = np.array(self.dataset.targets)

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        # divide and assign 2 shards/client
        for i in range(self.n_clients):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]),
                    axis=0,
                )

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