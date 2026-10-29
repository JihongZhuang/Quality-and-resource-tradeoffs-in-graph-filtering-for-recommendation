"""Read LightGCN-format interaction text and validate binary CSR data."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class DatasetBundle:
    """User-item matrices, held-out positives and dataset metadata."""
    name: str
    R_train: sp.csr_matrix          # 训练交互（二值），shape = (n_users, n_items)
    test_pos: dict[int, list[int]]  # 测试正样本：user -> [item, ...]
    n_users: int
    n_items: int
    card: dict = field(default_factory=dict)  # 数据卡：规模、度分布、零度节点等


def load_lightgcn_format(raw_dir: str, name: str) -> DatasetBundle:
    """读取 LightGCN 处理版目录，返回 CSR 训练图与测试正例。

    raw_dir 需包含 train.txt / test.txt（格式：每行 user 后接物品列表，空格分隔）。
    """
    def read_edges(path):
        users, items = [], []
        pos: dict[int, list[int]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                u = int(parts[0])
                its = [int(x) for x in parts[1:]]
                pos[u] = its
                users.extend([u] * len(its))
                items.extend(its)
        return np.asarray(users, dtype=np.int64), np.asarray(items, dtype=np.int64), pos

    tr_u, tr_i, _ = read_edges(os.path.join(raw_dir, "train.txt"))
    te_u, te_i, test_pos = read_edges(os.path.join(raw_dir, "test.txt"))

    n_users = int(max(tr_u.max(), te_u.max())) + 1
    n_items = int(max(tr_i.max(), te_i.max())) + 1

    # 训练图二值化：同一 (user, item) 重复边合并为 1
    R_train = sp.csr_matrix(
        (np.ones_like(tr_u), (tr_u, tr_i)), shape=(n_users, n_items), dtype=np.float64
    )
    R_train.sum_duplicates()
    R_train.data[:] = 1.0

    bundle = DatasetBundle(name=name, R_train=R_train, test_pos=test_pos,
                           n_users=n_users, n_items=n_items)
    bundle.card = make_data_card(bundle, n_train_pairs=int(R_train.nnz),
                                 n_test_pairs=int(len(te_u)))
    return bundle


def make_data_card(bundle: DatasetBundle, n_train_pairs: int, n_test_pairs: int) -> dict:
    """数据卡（计划书 §4.2）：三个层次的规模、密度、度分布与零度节点。"""
    R = bundle.R_train
    user_deg = np.asarray(R.sum(axis=1)).ravel()
    item_deg = np.asarray(R.sum(axis=0)).ravel()

    def q(x, p):
        return float(np.percentile(x, p)) if x.size else 0.0

    return {
        "name": bundle.name,
        "n_users": bundle.n_users,
        "n_items": bundle.n_items,
        "n_train": n_train_pairs,
        "n_test": n_test_pairs,
        "density_train": n_train_pairs / (bundle.n_users * bundle.n_items),
        "user_degree_median": q(user_deg, 50),
        "user_degree_p90": q(user_deg, 90),
        "user_degree_p99": q(user_deg, 99),
        "item_degree_median": q(item_deg, 50),
        "item_degree_p90": q(item_deg, 90),
        "item_degree_p99": q(item_deg, 99),
        "zero_degree_users_in_train": int((user_deg == 0).sum()),
        "zero_degree_items_in_train": int((item_deg == 0).sum()),
    }


def check_no_leakage(R_train: sp.csr_matrix, test_pos: dict[int, list[int]]) -> int:
    """泄漏检查：同一 (user, item) 不得同时出现在训练与测试。返回冲突边数。"""
    conflicts = 0
    R_coo = R_train.tocoo()
    train_set = set(zip(R_coo.row.tolist(), R_coo.col.tolist()))
    for u, its in test_pos.items():
        for i in its:
            if (u, i) in train_set:
                conflicts += 1
    return conflicts


def sample_unseen_items(
    R_train: sp.csr_matrix, users: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """为每个用户均匀采样一个训练集中未观测的物品。"""
    R = R_train.tocsr()
    # PyTorch ``Tensor.numpy()`` may expose a read-only view; SciPy sparse
    # advanced indexing mutates array flags internally, so own a writable copy.
    users = np.array(users, dtype=np.int64, copy=True)
    if users.ndim != 1 or np.any((users < 0) | (users >= R.shape[0])):
        raise ValueError("users must be a valid one-dimensional user-index array")
    if np.any(np.diff(R.indptr)[users] >= R.shape[1]):
        raise ValueError("cannot sample an unseen item for a user who observed every item")

    items = rng.integers(0, R.shape[1], size=users.size, dtype=np.int64)
    while users.size:
        seen = np.asarray(R[users, items]).reshape(-1) != 0
        if not seen.any():
            return items
        items[seen] = rng.integers(0, R.shape[1], size=int(seen.sum()), dtype=np.int64)
    return items
