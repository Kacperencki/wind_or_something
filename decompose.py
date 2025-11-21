# decompose.py

import time
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker
from tensorly.cp_tensor import cp_to_tensor
from tensorly.tucker_tensor import tucker_to_tensor
import torch

from config import CP_RANKS, TUCKER_RANKS


# Use PyTorch backend (can run on CPU or GPU)
tl.set_backend("pytorch")

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")


def _to_device(x_np: np.ndarray):
    """Convert NumPy array to torch tensor on GPU if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
    return x_t, device


def cp_decompose(X_np: np.ndarray, rank: int,
                 n_iter_max: int = 50, tol: float = 1e-4):
    """
    X_np: (D, T, F) as NumPy
    Returns:
        X_rec_np: reconstruction as NumPy
        rel_error: ||X - X_rec||_F / ||X||_F (float)
    """
    start = time.time()

    X_t, device = _to_device(X_np)

    # CP on torch tensor (GPU if available)
    weights, factors = parafac(
        X_t,
        rank=rank,
        init="random",
        n_iter_max=n_iter_max,
        tol=tol,
        normalize_factors=False,
    )

    X_rec_t = cp_to_tensor((weights, factors))

    num = tl.norm(X_t - X_rec_t)
    den = tl.norm(X_t)
    rel_error = (num / den).item()

    X_rec_np = X_rec_t.detach().cpu().numpy()

    elapsed = time.time() - start
    print(f"CP rank {rank}: rel_error={rel_error:.4f}, time={elapsed:.2f}s, device={device}")

    return X_rec_np, rel_error


def tucker_decompose(X_np: np.ndarray, ranks,
                     n_iter_max: int = 50, tol: float = 1e-4):
    """
    ranks: (R1, R2, R3)
    """
    start = time.time()

    X_t, device = _to_device(X_np)

    core, factors = tucker(
        X_t,
        rank=ranks,
        init="random",
        n_iter_max=n_iter_max,
        tol=tol,
    )

    X_rec_t = tucker_to_tensor((core, factors))

    num = tl.norm(X_t - X_rec_t)
    den = tl.norm(X_t)
    rel_error = (num / den).item()

    X_rec_np = X_rec_t.detach().cpu().numpy()

    elapsed = time.time() - start
    print(f"Tucker ranks {ranks}: rel_error={rel_error:.4f}, time={elapsed:.2f}s, device={device}")

    return X_rec_np, rel_error


def run_all_decompositions(X_np: np.ndarray):
    results_cp = {}
    for r in CP_RANKS:
        X_rec, err = cp_decompose(X_np, r)
        results_cp[r] = (X_rec, err)

    results_tucker = {}
    for ranks in TUCKER_RANKS:
        X_rec, err = tucker_decompose(X_np, ranks)
        results_tucker[ranks] = (X_rec, err)

    return results_cp, results_tucker


if __name__ == "__main__":
    from preprocess import load_dataset

    X, Y, _, _, _ = load_dataset()

    # you can limit days while testing if you want
    # X = X[:180]
    print("X shape:", X.shape)

    cp_res, tucker_res = run_all_decompositions(X)

    print("CP errors:")
    for r, (_, e) in cp_res.items():
        print("  rank", r, "->", e)

    print("Tucker errors:")
    for ranks, (_, e) in tucker_res.items():
        print("  ranks", ranks, "->", e)
