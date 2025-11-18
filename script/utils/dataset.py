import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .featurizer import protein_to_features

def get_loader(dataset, batchsize, num_workers = min(8, os.cpu_count()), shuffle=False):
    collator = GraphCollator(batchsize)
    batch_loader = DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        collate_fn=collator
    )
    return batch_loader

class GraphDataset(Dataset):
    """
    A PyTorch Dataset for protein graph representations.
    Each sample corresponds to one protein structure file (e.g., PDB),
    which is converted into node, edge, adjacency, label, and mask tensors.
    
    Supports caching processed tensors to .pt files to speed up repeated loading.
    Optionally adds Gaussian noise to node and edge features during training.
    """
    def __init__(self, datalist, nneighbor=20, noise=0.0, cache_dir=None):
        """
        Parameters:
        datalist (list): List of protein structure file paths (e.g., PDB files).
        nneighbor (int): Number of nearest neighbors used in graph construction.
        noise (float): Standard deviation of Gaussian noise added to node/edge features.
        cache_dir (str, optional): Directory to store cached .pt feature files.
        train (bool): Whether the dataset is for training (enables data augmentation).
        """
        self.datalist = datalist
        self.nneighbor = nneighbor
        self.noise = noise
        self.cache_dir = cache_dir

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.datalist)

    def __getitem__(self, idx):
        """Load backbone sample and return its graph representation."""
        infile = self.datalist[idx]
        if self.cache_dir is not None:
            cache_file = os.path.join(self.cache_dir, os.path.basename(infile) + ".pt")
            if os.path.exists(cache_file):
                data = torch.load(cache_file, map_location="cpu", weights_only=True)
                node, edgemat, adjmat, label, mask = data
            else:
                node, edgemat, adjmat, label, mask = self._process_feature(infile)
                data = node, edgemat, adjmat, label, mask
                torch.save(data, cache_file)
        else:
            node, edgemat, adjmat, label, mask = self._process_feature(infile)
        if self.noise >= 0:
            node = node + self.noise * torch.randn_like(node)
            edgemat = edgemat + self.noise * torch.randn_like(edgemat)
        return node, edgemat, adjmat, label, mask, infile
    
    def _process_feature(self, infile):
        """Convert a protein structure file into graph features from PDB files."""
        node, edgemat, adjmat, label, mask, _ = protein_to_features(infile, file_format='pdb', nneighbor=self.nneighbor, device='cpu')
        return node, edgemat, adjmat, label, mask

class GraphCollator:
    """
    Collate variable-length protein graphs into a batch using while loop accumulation.
    Each sample: (dat1, dat2, dat3, target, mask, name)
    """

    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        self.cache = []

    def __call__(self, batch):
        self.cache.extend(batch)

        if len(self.cache) >= self.batch_size:
            return self._stack_cache()

        return self._stack_cache_if_last()

    def _stack_cache_if_last(self):
        if len(self.cache) == 0:
            return None
        return self._stack_cache()

    def _stack_cache(self):
        num_samples = 0
        dat1_acc, dat2_acc, dat3_acc, target_acc, mask_acc, name_acc = None, None, None, [], [], []

        while num_samples < self.batch_size and len(self.cache) > 0:
            dat1, dat2, dat3, target, mask, name = self.cache.pop(0)
            num_samples += 1

            if dat1_acc is None:
                dat1_acc = dat1
                dat2_acc = dat2
                dat3_acc = dat3
            else:
                # cat node features
                dat1_acc = torch.cat([dat1_acc, dat1], dim=0)
                # cat edge features and adjacency
                dat2_acc = self.mat_connect(dat2_acc, dat2)
                dat3_acc = self.mat_connect(dat3_acc, dat3)

            target_acc.append(target)
            mask_acc.append(mask)
            name_acc.append(str(name))

        target_acc = torch.cat(target_acc, dim=0)
        mask_acc = torch.cat(mask_acc, dim=0)
        name_acc = "_".join(name_acc)

        return dat1_acc, dat2_acc, dat3_acc, target_acc, mask_acc, name_acc, num_samples

    @staticmethod
    def mat_connect(mat1, mat2):
        """
        Concatenate two adjacency/edge matrices into a block matrix.
        Supports:
        - mat1/mat2: node adjacency matrix (N,N) -> returns (M+N, M+N)
        - mat1/mat2: edge features (N,N,C) -> returns (M+N, M+N,C)
        """
        if mat1.dim() == 3 and mat2.dim() == 3:
            M, _, C1 = mat1.shape
            N, _, C2 = mat2.shape
            C = max(C1, C2)

            if C1 < C:
                mat1 = torch.cat([mat1, torch.zeros(M, M, C - C1, dtype=mat1.dtype, device=mat1.device)], dim=2)
            if C2 < C:
                mat2 = torch.cat([mat2, torch.zeros(N, N, C - C2, dtype=mat2.dtype, device=mat1.device)], dim=2)

            blank1 = torch.zeros(M, N, C, dtype=mat1.dtype, device=mat1.device)
            blank2 = torch.zeros(N, M, C, dtype=mat2.dtype, device=mat1.device)

            top = torch.cat([mat1, blank1], dim=1)
            bottom = torch.cat([blank2, mat2], dim=1)
            return torch.cat([top, bottom], dim=0)

        elif mat1.dim() == 2 and mat2.dim() == 2:
            M, _ = mat1.shape
            N, _ = mat2.shape
            top = torch.cat([mat1, torch.zeros(M, N, dtype=mat1.dtype, device=mat1.device)], dim=1)
            bottom = torch.cat([torch.zeros(N, M, dtype=mat2.dtype, device=mat2.device), mat2], dim=1)
            return torch.cat([top, bottom], dim=0)

        else:
            raise ValueError(f"Incompatible dimensions: mat1 {mat1.shape}, mat2 {mat2.shape}")
