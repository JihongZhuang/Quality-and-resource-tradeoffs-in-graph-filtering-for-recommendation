"""Complete-user evaluation of streaming direct and pinned reference routes.

The separately acquired Mem-GF source supplies reference normalization, filter
parameters and the released predictor. See the repository README for setup.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE.parent / "configs/reference.json").read_text(encoding="utf-8"))
ROUTES = ("author_snapshot", "streaming_direct")
METRICS = ("recall10", "recall20", "ndcg10", "ndcg20")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def author_namespace(directory):
    """AST locates trusted original segments; no source replacement or repair."""
    env = {"np": np, "sp": sp, "torch": torch}
    segments = {}
    for filename, names in MANIFEST["author_snapshot"]["required_definitions"].items():
        path = directory / filename
        expected = MANIFEST["author_snapshot"]["files"][filename]["sha256"]
        if sha(path) != expected:
            raise ValueError(f"Author source SHA-256 mismatch: {path}; use the pinned commit unchanged")
        source = path.read_text(encoding="utf-8")
        found = {}
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                if node.name in found:
                    raise ValueError(f"Duplicate definition: {node.name}")
                found[node.name] = ast.get_source_segment(source, node)
        if set(found) != set(names):
            raise ValueError(f"Missing author definitions in {filename}")
        for name, segment in found.items():
            exec(compile(segment, str(path), "exec"), env)
            segments[name] = hashlib.sha256(segment.encode()).hexdigest()
    return env, segments


def streaming_direct(model, x):
    # Classical direct_torch branch of the frozen helper.score, same operations.
    b = model.R_norm
    def action(z):
        return torch.sparse.mm(b.t(), torch.sparse.mm(b, z))
    coefficients = model.poly_coeff
    z = x.t()
    out = torch.zeros_like(z)
    for coefficient in coefficients:
        z = action(z)
        out += coefficient * z
    return out.t()


def canonical_top(pred, train, k=20):
    # Reused frozen helper: include all boundary ties, then score / item ID.
    pred = pred.numpy().copy() if isinstance(pred, torch.Tensor) else pred.copy()
    rr, cc = train.nonzero()
    pred[rr, cc] = -np.inf
    assert np.isfinite(pred[np.isfinite(pred)]).all() and not np.isnan(pred).any()
    top = np.empty((len(pred), k), dtype=np.int64)
    for i, row in enumerate(pred):
        boundary = np.partition(row, len(row) - k)[len(row) - k]
        candidates = np.flatnonzero(row >= boundary)
        order = np.lexsort((candidates, -row[candidates]))[:k]
        top[i] = candidates[order]
    return top


def metrics(top, test):
    gathered = test[np.arange(len(top))[:, None], top]
    hits = (gathered.toarray() if sp.issparse(gathered) else np.asarray(gathered)) > 0
    positives = np.diff(test.indptr)
    result = {}
    for k in (10, 20):
        discount = 1 / np.log2(np.arange(2, k + 2))
        idcg = np.r_[0., np.cumsum(discount)][np.minimum(positives, k)]
        result[f'recall{k}'] = hits[:, :k].sum(axis=1) / np.maximum(positives, 1)
        result[f'ndcg{k}'] = (hits[:, :k] * discount).sum(axis=1) / np.maximum(idcg, 1e-30)
    return result


class CountSpMM:
    def __enter__(self):
        self.original, self.calls = torch.sparse.mm, 0
        def counted(*args, **kwargs):
            self.calls += 1
            return self.original(*args, **kwargs)
        torch.sparse.mm = counted
        return self

    def __exit__(self, *args):
        torch.sparse.mm = self.original


def model_for(namespace, raw, config):
    return namespace["Mem_GF"](SimpleNamespace(power=config["power"]), raw,
        config["filter"], namespace["get_polynomial_coeffs"](config["filter"]), torch.device("cpu"))


def check_data(train, test):
    if train.shape != test.shape or not train.shape[0] or train.shape[1] < 20:
        raise ValueError("Train/test need equal nonempty user-item shapes and at least 20 items")
    for label, matrix in (("train", train), ("test", test)):
        if not matrix.has_canonical_format or not np.all(matrix.data == 1):
            raise ValueError(f"{label} must be canonical binary CSR (no duplicates/explicit zeros)")
    degree = np.diff(train.indptr)
    if np.any(degree == 0):
        raise ValueError("Zero train rows trigger original author NaNs; no silent repair is allowed")
    if np.any(train.shape[1] - degree < 20):
        raise ValueError("Every user needs at least 20 unseen candidate items")


def self_check(namespace):
    torch.manual_seed(20261019)
    raw = (torch.rand(17, 25) / 12).to_sparse().coalesce()
    x = torch.rand(3, 25)
    counts = {}
    with torch.no_grad():
        for degree in (2, 5):
            model = model_for(namespace, raw, {"filter": degree, "power": .9})
            with CountSpMM() as counted:
                author = model.predict_batch(x)
            assert counted.calls == 2 * (degree + 1)
            with CountSpMM() as counted_direct:
                direct = streaming_direct(model, x)
            assert counted_direct.calls == 2 * degree
            p = model.R_norm.to_dense().t() @ model.R_norm.to_dense()
            current, expected = x, torch.zeros_like(x)
            for coefficient in model.poly_coeff:
                current = current @ p
                expected += coefficient * current
            assert torch.allclose(direct, expected, rtol=1e-4, atol=1e-5)
            assert not torch.allclose(author, direct, rtol=1e-4, atol=1e-6)
            norms = torch.norm(x, dim=1, keepdim=True)
            q, alpha, beta = namespace["lanczos_algorithm"](model.R_norm, x / norms,
                degree + 1, torch.device("cpu"))
            ft = namespace["polynomial_filter"](namespace["compute_T"](alpha, beta,
                torch.device("cpu")), model.poly_coeff, torch.device("cpu"))
            e1 = torch.zeros((len(x), degree + 1, 1))
            e1[:, 0, 0] = 1
            mirrored = norms * torch.bmm(torch.bmm(ft, e1).transpose(1, 2),
                q[:-1].permute(1, 0, 2)).squeeze(1)
            assert torch.equal(author, mirrored)
            assert torch.isnan(model.predict_batch(torch.zeros_like(x))).all()
            assert torch.equal(streaming_direct(model, torch.zeros_like(x)), torch.zeros_like(x))
            counts[str(degree)] = {"author_snapshot": counted.calls,
                                  "streaming_direct": counted_direct.calls}
        alpha = torch.zeros(3, 1, 1)
        beta = torch.tensor([11., 22., 33.]).reshape(3, 1, 1)
        t = namespace["compute_T"](alpha, beta, torch.device("cpu"))
        assert t[0, 0, 1] == 22 and t[0, 1, 2] == 33
        train = sp.csr_matrix(([1.], ([0], [0])), shape=(2, 24))
        test = sp.csr_matrix(([1., 1.], ([0, 0], [1, 22])), shape=(2, 24))
        top = canonical_top(np.ones((2, 24), np.float32), train)
        assert np.array_equal(top[0], np.arange(1, 21)) and np.array_equal(top[1], np.arange(20))
        result = metrics(top, test)
        assert np.array_equal(result["recall20"], [.5, 0.])
        assert np.allclose(result["ndcg20"], [1 / (1 + 1 / np.log2(3)), 0.])
        try:
            check_data(train, test)
        except ValueError as exc:
            assert "Zero train rows" in str(exc)
        else:
            raise AssertionError("Zero training row must be rejected")
    return {"passed": True, "synthetic_only": True, "actual_SpMM_per_batch": counts,
            "author_beta_offset_retained": True, "author_equals_original_segments": True,
            "direct_matches_dense_polynomial": True, "tie_mask_empty_test_checked": True,
            "timing_or_speedup_conclusion": None}


def replay(args, namespace, segments):
    train = sp.load_npz(args.train).tocsr()
    test = sp.load_npz(args.test).tocsr()
    check_data(train, test)
    config = MANIFEST["configs"][args.config]
    normalized, di, di_inv = namespace["normalize_sparse_adjacency_matrix_sparse"](train, config["gamma"])
    raw = namespace["csr2torch"](normalized, torch.device("cpu")).coalesce()
    model = model_for(namespace, raw, config)
    del normalized, di, di_inv, raw
    users = np.arange(train.shape[0], dtype=np.int64)
    top20 = np.empty((len(users), 20), dtype=np.int64)
    row_hashes = np.empty(len(users), dtype="S64")
    per_user = {key: np.empty(len(users), dtype=np.float64) for key in METRICS}
    batches = []
    with torch.no_grad(), CountSpMM() as counted:
        for start in range(0, len(users), args.batch_size):
            end = min(start + args.batch_size, len(users))
            tr, ts = train[start:end], test[start:end]
            x = torch.from_numpy(tr.toarray().astype(np.float32))
            before = counted.calls
            score = model.predict_batch(x) if args.route == "author_snapshot" else streaming_direct(model, x)
            expected = 2 * (config["filter"] + (args.route == "author_snapshot"))
            if counted.calls - before != expected or not torch.isfinite(score).all():
                raise RuntimeError("Unexpected sparse action count or nonfinite original output; no repair applied")
            matrix = score.contiguous().numpy()
            row_hashes[start:end] = [hashlib.sha256(memoryview(row)).hexdigest().encode("ascii") for row in matrix]
            top20[start:end] = canonical_top(score, tr)
            values = metrics(top20[start:end], ts)
            for key in METRICS:
                per_user[key][start:end] = values[key]
            batches.append({"start": start, "end": end, "actual_SpMM": counted.calls - before,
                "full_score_sha256": hashlib.sha256(memoryview(matrix)).hexdigest()})
            del score, matrix, x, tr, ts
    input_records = {label: {"name": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
                     for label, path in (("train", args.train), ("test", args.test))}
    reference = MANIFEST["reference_inputs"][config["dataset"]]
    input_match = all(input_records[key]["sha256"] == reference[key]["sha256"] for key in input_records)
    args.output.mkdir(parents=True, exist_ok=False)
    witness = args.output / "per_user.npz"
    np.savez_compressed(witness, users=users, top20=top20, full_score_sha256_per_user=row_hashes, **per_user)
    summary = {"status": "complete", "purpose": "correctness replay; not a formal timing repetition",
        "route": args.route, "config": args.config, "parameters": config, "Q": 1,
        "users": len(users), "user_order": "all IDs ascending, no sampling", "shape": list(train.shape),
        "train_nnz": train.nnz, "empty_test_users_retained": int(np.sum(np.diff(test.indptr) == 0)),
        "batch_size": args.batch_size, "actual_SpMM": counted.calls, "batches": batches,
        "metrics": {key: float(per_user[key].mean()) for key in METRICS},
        "inputs": input_records, "exact_reference_input_hashes_match": input_match,
        "reference_input_and_batch_match": input_match and args.batch_size == 64,
        "author_commit": MANIFEST["author_snapshot"]["commit"], "author_segment_sha256": segments,
        "script_sha256": sha(__file__), "manifest_sha256": sha(HERE.parent / "configs/reference.json"),
        "per_user_sha256": sha(witness), "environment": {"python": platform.python_version(),
            "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
            "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(), "device": "cpu", "dtype": "float32"},
        "timing_or_speedup_conclusion": None}
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")
    return {key: summary[key] for key in ("status", "route", "config", "users", "metrics",
        "exact_reference_input_hashes_match", "timing_or_speedup_conclusion")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-source", required=True, type=Path)
    parser.add_argument("--self-check", action="store_true", help="only tiny synthetic matrices; no real data read")
    parser.add_argument("--config", choices=tuple(MANIFEST["configs"]))
    parser.add_argument("--route", choices=ROUTES)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--output", type=Path, help="new directory; existing paths are never overwritten")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1:
        parser.error("--threads and --batch-size must be positive")
    if not args.self_check and any(getattr(args, key) is None for key in ("config", "route", "train", "test", "output")):
        parser.error("replay requires --config, --route, --train, --test, --output")
    if args.self_check and any(getattr(args, key) is not None for key in ("config", "route", "train", "test", "output")):
        parser.error("--self-check accepts only --author-source, --threads, --batch-size")
    if args.output is not None and args.output.exists():
        parser.error("--output already exists; use a fresh directory")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    global np, sp, torch, scipy
    import numpy as np
    import scipy
    import scipy.sparse as sp
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    namespace, segments = author_namespace(args.author_source)
    result = self_check(namespace) if args.self_check else replay(args, namespace, segments)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
