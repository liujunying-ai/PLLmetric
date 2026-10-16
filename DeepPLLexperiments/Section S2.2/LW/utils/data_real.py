import os
import numpy as np
import torch
import scipy.sparse as sp
from scipy.io import loadmat
from torch.utils.data import Dataset


def _extract_fold(mat, key, fold=0):
    """从 MATLAB cell 数组里提取第 fold 折的索引（fold 从 0 开始）。"""
    arr = mat[key]
    idx = arr[0, fold]
    idx = np.asarray(idx).flatten().astype(np.int64)
    return idx - 1


class RealPartialDataset(Dataset):
    """
    加载真实偏标记数据集（.mat 格式）。
    优先使用 .mat 里的 index_train / index_test（指定 fold）。
    """
    def __init__(self, root, dataset_name, train_or_not=True,
                 split_ratio=0.7, seed=42, fold=0, use_predefined_split=True):
        self.dataset_name = dataset_name
        self.train = train_or_not

        mat_path = os.path.join(root, f'{dataset_name}.mat')
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f'找不到文件: {mat_path}')
        mat = loadmat(mat_path)

        features = mat['data'].astype(np.float32)
        target = mat['target'].T
        partial = mat['partial_target'].T

        if sp.issparse(target):
            target = target.toarray()
        if sp.issparse(partial):
            partial = partial.toarray()
        target = target.astype(np.int64)
        partial = partial.astype(np.float32)

        n_samples = features.shape[0]
        self.num_features = features.shape[1]
        self.num_classes = target.shape[1]

        if use_predefined_split and ('index_train' in mat.keys()) and ('index_test' in mat.keys()):
            train_idx = _extract_fold(mat, 'index_train', fold=fold)
            test_idx = _extract_fold(mat, 'index_test', fold=fold)
            print(f'[{dataset_name}] 使用预定义的 fold {fold}: '
                  f'train={len(train_idx)}, test={len(test_idx)}')
        else:
            rng = np.random.RandomState(seed)
            indices = rng.permutation(n_samples)
            split = int(split_ratio * n_samples)
            train_idx = indices[:split]
            test_idx = indices[split:]
            print(f'[{dataset_name}] 使用随机划分 70/30 (seed={seed})')

        # Z-score 归一化（用训练集统计量）
        train_mean = features[train_idx].mean(axis=0)
        train_std = features[train_idx].std(axis=0) + 1e-8
        features = (features - train_mean) / train_std

        idx = train_idx if self.train else test_idx

        self.data = torch.from_numpy(features[idx]).float()
        self.partial = torch.from_numpy(partial[idx]).float()
        self.true_labels = torch.from_numpy(target[idx].argmax(axis=1)).long()

        row_sum = self.partial.sum(dim=1, keepdim=True)
        row_sum[row_sum == 0] = 1.0
        self.partial = self.partial / row_sum

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return (self.data[idx], self.partial[idx],
                self.true_labels[idx], idx)