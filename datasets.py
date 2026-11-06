import torch
import numpy as np
from PIL import Image
import random
import torchvision
import torchvision.transforms as transforms

from collections import defaultdict
import functools
import json
import os

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
OOV = "<OOV>"

ALL_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,;:'!?\"()[]{}<>-_/\\\n\t"
)

VOCAB = [PAD, BOS, EOS, OOV] + list(ALL_CHARS)
VOCAB_SIZE = len(VOCAB)

CHAR2IDX = {c: i for i, c in enumerate(VOCAB)}

SEQ_LEN = 80 
NUM_CLASSES = {'CIFAR10': 10, 'CIFAR100': 100, 'FEMNIST': 62, 'Shakespeare': VOCAB_SIZE}

def read_dir(data_dir):
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data

def read_data(train_data_dir, test_data_dir):

    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data
    
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@functools.lru_cache(maxsize=None)
def _get_leaf_data(name):
    root = os.path.join(_BASE_DIR, f"{name}_data")
    return read_data(os.path.join(root, "train"), os.path.join(root, "test"))

@functools.lru_cache(maxsize=None)
def _get_cifar(name, train):
    raw = getattr(torchvision.datasets, name)(root=os.path.join(_BASE_DIR, 'cifar_data'), train=train, download=True)
    return raw.data, np.array(raw.targets)


_CIFAR = ('CIFAR10', 'CIFAR100')
_CIFAR10_STATS = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

def transform_dataset(is_train, dataset):
    if dataset == 'FEMNIST':
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if dataset not in _CIFAR:
        return None  
    steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if is_train else []
    steps.append(transforms.ToTensor())
    if dataset == 'CIFAR10':
        steps.append(transforms.Normalize(*_CIFAR10_STATS))
    return transforms.Compose(steps)

def _dirichlet_partition(labels, client_count, min_require_size, beta):
    n_train = len(labels)
    min_size = 0
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(client_count)]
        for k in range(labels.max() + 1):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, client_count))
            proportions = np.array([p * (len(idx_j) < n_train / client_count) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min(len(idx_j) for idx_j in idx_batch)

    for idx_j in idx_batch:
        np.random.shuffle(idx_j)
    return dict(enumerate(idx_batch))

def find_distribution_map(dataset, partition, client_count, batch_size, beta=0.1):
    if dataset in _CIFAR:
        labels = _get_cifar(dataset, True)[1]
        if partition == 'iid':
            return dict(enumerate(np.array_split(np.random.permutation(len(labels)), client_count)))
        if partition == 'noniiddir':
            return _dirichlet_partition(labels, client_count, batch_size, beta)
        raise ValueError(f"Unknown partition for {dataset}: {partition}")

    if dataset in ('FEMNIST', 'Shakespeare') and partition != 'noniiddir':
        raise ValueError(f"{dataset} only supports partition 'noniiddir', got {partition}")
    if dataset == 'FEMNIST':
        eligible = range(len(_get_leaf_data('femnist')[0]))
    elif dataset == 'Shakespeare':
        train_clients, _, train_data, _ = _get_leaf_data('shakespeare')

        eligible = [j for j, user in enumerate(train_clients) if len(train_data[user]['x']) >= batch_size]

        if len(eligible) < client_count:
            raise ValueError(
                f"only {len(eligible)} roles have at least {batch_size} samples, "
                f"but {client_count} clients were requested"
            )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    net_dataidx_map = dict(enumerate(random.sample(eligible, client_count)))
    print(net_dataidx_map)
    return net_dataidx_map


def _load_cifar(name, idxs, train):
    data, target = _get_cifar(name, train)
    if idxs is not None:
        data, target = data[idxs], target[idxs]
    return data, target

def _load_femnist(writer_idx, train):
    """One writer's training samples, or the test samples of every writer.

    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    train_clients, _, train_data, test_data = _get_leaf_data('femnist')
    users = [train_data[train_clients[writer_idx]]] if train else test_data.values()
    data = np.concatenate([np.asarray(u['x'], dtype=np.float32).reshape(-1, 28, 28) for u in users])
    target = np.concatenate([np.asarray(u['y']) for u in users])
    return data, target


class LocalDataset(torch.utils.data.Dataset):

    def __init__(self, data, target, num_classes, transform=None):
        self.data = data
        self.target = target
        self.num_classes = num_classes
        self.transform = transform

    def fliplabels(self):
        self.target = (self.num_classes - 1) - self.target

    def __getitem__(self, index):
 
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


def _encode_chars(text):
    codes = np.frombuffer(text.encode('utf-32-le'), dtype='<u4')

    lut = np.full(max(map(ord, ALL_CHARS)) + 2, CHAR2IDX[OOV], dtype=np.int64)
    for c in ALL_CHARS:
        lut[ord(c)] = CHAR2IDX[c]
    return lut[np.minimum(codes, len(lut) - 1)]


class Shakespeare_local(torch.utils.data.Dataset):

    def __init__(self, client_id=None, train=True):
        self.client_id = client_id
        self.train = train
        self.data, self.target = self._build_dataset()

    @staticmethod
    def _encode(users):
        xs = [x for u in users for x in u['x']]
        ys = [y for u in users for y in u['y']]
        assert len(xs) == len(ys)
        assert all(len(x) == SEQ_LEN for x in xs) and all(len(y) == 1 for y in ys)
        return _encode_chars(''.join(xs)).reshape(len(xs), SEQ_LEN), _encode_chars(''.join(ys))

    def _build_dataset(self):
        train_clients, _, train_data, test_data = _get_leaf_data('shakespeare')
        if self.train:
            X_arr, Y_arr = self._encode([train_data[train_clients[self.client_id]]])
            print(f"[INFO] Client {self.client_id} | X shape: {X_arr.shape}, Y shape: {Y_arr.shape}")
            print(f"[INFO] Client {self.client_id} | OOV fraction: {(X_arr == CHAR2IDX[OOV]).mean():.6f}")
        else:
            X_arr, Y_arr = self._encode([test_data[user] for user in train_clients])
            print(f"[INFO] Test set | X shape: {X_arr.shape}, Y shape: {Y_arr.shape}")

        return X_arr, Y_arr

    def fliplabels(self):
        self.target = (VOCAB_SIZE - 1) - self.target

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index], dtype=torch.long),
            torch.tensor(self.target[index], dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)


_CROP_PAD = 4  
class DeviceBatches:

    def __init__(self, dataset, name, batch_size, device):
        if len(dataset) < batch_size:
            raise ValueError(f"client has {len(dataset)} samples, fewer than one batch of {batch_size}")
        self.dataset = dataset
        self.name = name
        self.batch_size = batch_size
        self.device = device
        self.n = len(dataset)

        if name in _CIFAR:
            data = torch.as_tensor(dataset.data).permute(0, 3, 1, 2)
            self.size = data.shape[-1]
            self.data = torch.nn.functional.pad(data, (_CROP_PAD,) * 4).contiguous().to(device)
            self._channels = torch.arange(3, device=device).view(1, 3, 1, 1)
            self._span = torch.arange(self.size, device=device)
            if name == 'CIFAR10':
                self._mean = torch.tensor(_CIFAR10_STATS[0], device=device).view(1, 3, 1, 1)
                self._std = torch.tensor(_CIFAR10_STATS[1], device=device).view(1, 3, 1, 1)
        elif name == 'FEMNIST':
            data = torch.as_tensor(dataset.data, dtype=torch.float32).unsqueeze(1)
            self.data = ((data - 0.5) / 0.5).to(device)
        elif name == 'Shakespeare':
            self.data = torch.as_tensor(dataset.data, dtype=torch.long).to(device)
        else:
            raise ValueError(f"Unknown dataset: {name}")

        self._target_src = None
        self.target = None
        self._perm = None
        self._pos = self.n  

    def _sync_targets(self):
        if self.dataset.target is not self._target_src:
            self._target_src = self.dataset.target
            self.target = torch.as_tensor(np.asarray(self._target_src), dtype=torch.long).to(self.device)

    def _crop_flip(self, idx, offsets, flip):
        rows = offsets[:, 0:1] + self._span                               
        cols = offsets[:, 1:2] + self._span
        cols = torch.where(flip.unsqueeze(1), cols.flip(1), cols)
        x = self.data[idx.view(-1, 1, 1, 1), self._channels, rows[:, None, :, None], cols[:, None, None, :]]
        x = x.float().div_(255)                     
        if self.name == 'CIFAR10':
            x = x.sub_(self._mean).div_(self._std)
        return x

    def next(self):
        self._sync_targets()
        if self._pos + self.batch_size > self.n:
            self._perm = torch.randperm(self.n).to(self.device)
            self._pos = 0
        idx = self._perm[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size

        if self.name in _CIFAR:
            offsets = torch.randint(0, 2 * _CROP_PAD + 1, (self.batch_size, 2)).to(self.device)
            flip = (torch.rand(self.batch_size) < 0.5).to(self.device)
            data = self._crop_flip(idx, offsets, flip)
        else:
            data = self.data[idx]
        return data, self.target[idx]


def build_dataset(name, idx=None, train=True):
    transform = transform_dataset(train, name)
    if name == 'Shakespeare':
        return Shakespeare_local(idx, train)
    if name in _CIFAR:
        data, target = _load_cifar(name, idx, train)
    elif name == 'FEMNIST':
        data, target = _load_femnist(idx, train)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return LocalDataset(data, target, NUM_CLASSES[name], transform)

def print_data_distribution(clients):
    print("Data Distribution:")
    num_classes = NUM_CLASSES[clients[0].data_source]
    counts = np.stack([np.bincount(np.asarray(c.data_set.target).ravel(), minlength=num_classes)
                       for c in clients])
    with np.printoptions(threshold=np.inf, linewidth=200):
        print(counts)

