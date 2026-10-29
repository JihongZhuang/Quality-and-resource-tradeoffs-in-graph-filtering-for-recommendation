"""Data preparation helpers for binary user-item CSR matrices.

The main preparation entry is prepare_data.py; split helpers are also provided.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data import load_lightgcn_format, make_data_card


def split_train_valid(R_train: sp.csr_matrix, ratio: float, seed: int):
    """按用户随机抽 ratio 比例的训练边为验证集；每用户至少保留一条训练边。

    返回 (R_tr, R_va, n_users_could_not_split)。
    同一 (user,item) 不跨集合：验证边从训练边中移除。
    """
    rng = np.random.default_rng(seed)
    R = R_train.tocsr()
    indptr, indices = R.indptr, R.indices
    tr_rows, tr_cols, va_rows, va_cols = [], [], [], []
    n_could_not = 0
    for u in range(R.shape[0]):
        s, e = indptr[u], indptr[u + 1]
        items = indices[s:e]
        deg = e - s
        if deg == 0:
            continue
        if deg == 1:
            n_could_not += 1
            tr_rows.append(u); tr_cols.append(items[0])
            continue
        n_va = max(1, int(round(deg * ratio)))
        n_va = min(n_va, deg - 1)  # 至少保留一条训练边
        va_pick = rng.choice(deg, size=n_va, replace=False)
        mask = np.zeros(deg, dtype=bool)
        mask[va_pick] = True
        for it, is_va in zip(items, mask):
            if is_va:
                va_rows.append(u); va_cols.append(it)
            else:
                tr_rows.append(u); tr_cols.append(it)
    shape = R.shape
    R_tr = sp.csr_matrix((np.ones(len(tr_rows)), (tr_rows, tr_cols)), shape=shape, dtype=np.float64)
    R_va = sp.csr_matrix((np.ones(len(va_rows)), (va_rows, va_cols)), shape=shape, dtype=np.float64)
    return R_tr, R_va, n_could_not


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config, encoding="utf-8"))
    name = cfg["dataset"]
    t0 = time.time()

    bundle = load_lightgcn_format(cfg["raw_dir"], name)
    seed = cfg["split"]["seed"]
    ratio = cfg["split"]["valid_ratio_per_user"]
    R_tr, R_va, n_could_not = split_train_valid(bundle.R_train, ratio, seed)

    out_dir = os.path.join("data", "processed", name)
    os.makedirs(out_dir, exist_ok=True)
    sp.save_npz(os.path.join(out_dir, "R_train.npz"), R_tr)
    sp.save_npz(os.path.join(out_dir, "R_valid.npz"), R_va)
    sp.save_npz(os.path.join(out_dir, "R_test.npz"), _pos_to_csr(bundle.test_pos, R_tr.shape))
    sp.save_npz(os.path.join(out_dir, "R_train_full_official.npz"), bundle.R_train)

    # 评估用正例字典
    def pos_dict(R):
        d = {}
        for u in range(R.shape[0]):
            s, e = R.indptr[u], R.indptr[u + 1]
            if e > s:
                d[u] = R.indices[s:e].tolist()
        return d
    json.dump(pos_dict(R_va), open(os.path.join(out_dir, "valid_pos.json"), "w"))
    json.dump(bundle.test_pos, open(os.path.join(out_dir, "test_pos.json"), "w"))

    card = make_data_card(bundle, int(bundle.R_train.nnz), int(sum(len(v) for v in bundle.test_pos.values())))
    card.update({
        "split_protocol": cfg["split"]["protocol"],
        "valid_ratio": ratio, "seed": seed,
        "n_train_after_split": int(R_tr.nnz),
        "n_valid": int(R_va.nnz),
        "users_could_not_split": n_could_not,
        "config_sha256": hashlib.sha256(
            json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
    })
    json.dump(card, open(os.path.join(out_dir, "data_card.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # Mem-GF 作者代码对照格式（官方训练集全集，不含本项目验证划分）
    ext_dir = os.path.join("external", "Mem-GF", "dataset")
    os.makedirs(ext_dir, exist_ok=True)
    sp.save_npz(os.path.join(ext_dir, f"{name}_train.npz"),
                bundle.R_train.astype(np.float32).tocsr())
    sp.save_npz(os.path.join(ext_dir, f"{name}_test.npz"),
                _pos_to_csr(bundle.test_pos, R_tr.shape).astype(np.float32).tocsr())

    print(f"[{name}] 训练 {R_tr.nnz} / 验证 {R_va.nnz} / 测试 {card['n_test']}，"
          f"无法划出验证边的用户 {n_could_not}，耗时 {time.time()-t0:.1f}s")
    print(f"  写出 {out_dir}/ 与 external/Mem-GF/dataset/{name}_*.npz")


def _pos_to_csr(pos: dict, shape) -> sp.csr_matrix:
    rows, cols = [], []
    for u, its in pos.items():
        rows.extend([u] * len(its)); cols.extend(its)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape, dtype=np.float64)


if __name__ == "__main__":
    main()
