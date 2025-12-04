# decompose.py

import time
from typing import Dict, Tuple, Any

import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker
from tensorly.cp_tensor import cp_to_tensor
from tensorly.tucker_tensor import tucker_to_tensor
import torch
from sklearn.decomposition import PCA

from config import CP_RANKS, TUCKER_RANKS, SEED, PCA_K # PCA_K handled locally

# Use PyTorch backend (can run on CPU or GPU)
tl.set_backend("pytorch")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(SEED)



def _to_tl_tensor(X: np.ndarray):
    """Helper: NumPy -> TensorLy tensor on chosen device."""
    return tl.tensor(X, device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# CP DECOMPOSITION + LATENT FEATURES
# ---------------------------------------------------------------------------

def cp_decompose(
    X: np.ndarray,
    rank: int,
    n_iter_max: int = 200,
    tol: float = 1e-5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    CP decomposition of X (D,T,F) with given rank.

    Returns:
        X_hat: reconstructed tensor (D,T,F)
        info: dict with factors, weights, relative error, etc.
    """
    X_tl = _to_tl_tensor(X)

    start = time.time()
    weights, factors = parafac(
        X_tl,
        rank=rank,
        init="random",
        n_iter_max=n_iter_max,
        tol=tol,
    )
    elapsed = time.time() - start

    # Reconstruction and relative error
    X_hat_tl = cp_to_tensor((weights, factors))
    rel_error = float(
        tl.norm(X_tl - X_hat_tl) / tl.norm(X_tl)
    )

    # Move to CPU / NumPy
    X_hat = X_hat_tl.detach().cpu().numpy()
    weights_np = weights.detach().cpu().numpy()
    A = factors[0].detach().cpu().numpy()  # days
    B = factors[1].detach().cpu().numpy()  # time slots
    C = factors[2].detach().cpu().numpy()  # features

    info = {
        "rank": rank,
        "weights": weights_np,
        "A": A,
        "B": B,
        "C": C,
        "rel_error": rel_error,
        "time": elapsed,
        "device": str(device),
    }
    return X_hat, info


def cp_latent_features(X: np.ndarray, rank: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    CP latent features for each (day, slot):

        Z[d, t, r] = A[d, r] * B[t, r]

    where A and B are the day/time factor matrices from CP.

    Returns:
        Z:   (D, T, rank) latent tensor
        info: same dict as cp_decompose
    """
    _, info = cp_decompose(X, rank)
    A = info["A"]  # (D, R)
    B = info["B"]  # (T, R)
    D, R = A.shape
    T, _ = B.shape

    Z = np.zeros((D, T, R), dtype=np.float32)
    for r in range(R):
        Z[:, :, r] = np.outer(A[:, r], B[:, r])

    info["latent_dim"] = R
    return Z, info


# ---------------------------------------------------------------------------
# TUCKER DECOMPOSITION + LATENT FEATURES
# ---------------------------------------------------------------------------

def tucker_decompose(
    X: np.ndarray,
    ranks: Tuple[int, int, int],
    n_iter_max: int = 200,
    tol: float = 1e-5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Tucker decomposition of X (D,T,F) with ranks (R_D, R_T, R_F).

    Returns:
        X_hat: reconstructed tensor (D,T,F)
        info: dict with core, factor matrices, relative error, etc.
    """
    X_tl = _to_tl_tensor(X)

    start = time.time()
    # IMPORTANT: TensorLy uses 'rank=', not 'ranks='
    core, factors = tucker(
        X_tl,
        rank=ranks,
        init="random",
        n_iter_max=n_iter_max,
        tol=tol,
    )
    elapsed = time.time() - start

    X_hat_tl = tucker_to_tensor((core, factors))
    rel_error = float(
        tl.norm(X_tl - X_hat_tl) / tl.norm(X_tl)
    )

    # Move to CPU / NumPy
    X_hat = X_hat_tl.detach().cpu().numpy()
    core_np = core.detach().cpu().numpy()
    A_D = factors[0].detach().cpu().numpy()  # days
    A_T = factors[1].detach().cpu().numpy()  # time slots
    A_F = factors[2].detach().cpu().numpy()  # features

    info = {
        "ranks": tuple(ranks),
        "core": core_np,
        "A_D": A_D,
        "A_T": A_T,
        "A_F": A_F,
        "rel_error": rel_error,
        "time": elapsed,
        "device": str(device),
    }
    return X_hat, info


def tucker_latent_features_core(
    X: np.ndarray,
    ranks: Tuple[int, int, int],
    n_iter_max: int = 200,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Latent features via core contraction:

        X ≈ G ×1 A_D ×2 A_T ×3 A_F

    For each (d,t):

        Z[d, t, :] = sum_{i,j} A_D[d,i] * A_T[t,j] * G[i,j,:]

    Z: (D, T, R_F)
    """
    _, info = tucker_decompose(X, ranks=ranks,
                               n_iter_max=n_iter_max, tol=tol)

    G   = info["core"]   # (R_D, R_T, R_F)
    A_D = info["A_D"]    # (D,   R_D)
    A_T = info["A_T"]    # (T,   R_T)

    # einsum indices: i=R_D, j=R_T, k=R_F
    # A_D: di, A_T: tj, G: ijk  ->  Z: dtk
    Z = np.einsum("di,tj,ijk->dtk", A_D, A_T, G)

    info["latent_dim"] = G.shape[2]  # R_F
    return Z.astype(np.float32), info

def tucker_latent_features_mode3(
    X: np.ndarray,
    ranks: Tuple[int, int, int],
    n_iter_max: int = 200,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Latent features ala 'PCA on feature mode':

        X: (D, T, F)
        A_F: (F, R_F)

    Z[d, t, :] = X[d, t, :] @ A_F   ->  Z: (D, T, R_F)
    """
    _, info = tucker_decompose(X, ranks=ranks,
                               n_iter_max=n_iter_max, tol=tol)

    A_F = info["A_F"]                           # (F, R_F)
    # Contract feature mode with A_F:
    Z = np.tensordot(X, A_F, axes=([2], [0]))  # -> (D, T, R_F)

    info["latent_dim"] = A_F.shape[1]
    return Z.astype(np.float32), info


# ---------------------------------------------------------------------------
# PCA LATENT FEATURES
# ---------------------------------------------------------------------------

def pca_latent_features(X: np.ndarray, k: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    PCA baseline on matrix view:

        X_flat = (D*T, F)

    PCA -> (D*T, k) then reshape back to (D, T, k).
    """
    D, T, F = X.shape
    X_flat = X.reshape(D * T, F)

    pca = PCA(n_components=k, random_state=SEED)
    Z_flat = pca.fit_transform(X_flat)

    Z = Z_flat.reshape(D, T, k).astype(np.float32)
    info = {
        "k": k,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "latent_dim": k,
    }
    return Z, info


# ---------------------------------------------------------------------------
# HIGH-LEVEL DRIVER
# ---------------------------------------------------------------------------

def run_all_decompositions(X: np.ndarray):
    """
    Convenience wrapper used in experiments.

    Returns:
        cp_results:      {rank -> (X_hat, rel_error)}
        tucker_results:  {ranks_tuple -> (X_hat, rel_error)}
        cp_latent:       {rank -> Z_cp (D,T,rank)}
        pca_latent:      {k -> Z_pca (D,T,k)}

    NOTE: Tucker latent features are available via tucker_latent_features()
          and not returned here to keep the 4-tuple API.
    """
    cp_results: Dict[int, Tuple[np.ndarray, float]] = {}
    tucker_results: Dict[Tuple[int, int, int], Tuple[np.ndarray, float]] = {}
    cp_latent_dict: Dict[int, np.ndarray] = {}
    pca_latent_dict: Dict[int, np.ndarray] = {}

    # CP
    for rank in CP_RANKS:
        print(f"[CP] rank={rank}")
        X_hat, info = cp_decompose(X, rank)
        cp_results[rank] = (X_hat, info["rel_error"])
        Z_cp, _ = cp_latent_features(X, rank)
        cp_latent_dict[rank] = Z_cp

    # Tucker (reconstruction errors only here)
    for ranks in TUCKER_RANKS:
        print(f"[Tucker] ranks={ranks}")
        X_hat, info = tucker_decompose(X, ranks)
        tucker_results[tuple(ranks)] = (X_hat, info["rel_error"])

    # PCA
    for k in PCA_K:
        print(f"[PCA] k={k}")
        Z_pca, _ = pca_latent_features(X, k)
        pca_latent_dict[k] = Z_pca

    return cp_results, tucker_results, cp_latent_dict, pca_latent_dict


if __name__ == "__main__":
    # Quick smoke test if you run this file directly
    from preprocess import load_dataset

    X, Y, dates, mean, std = load_dataset()
    print("X shape:", X.shape)

    cp_res, tucker_res, cp_latent, pca_latent = run_all_decompositions(X)

    print("CP errors:")
    for r, (_, e) in cp_res.items():
        print("  rank", r, "->", e)

    print("Tucker errors:")
    for ranks, (_, e) in tucker_res.items():
        print("  ranks", ranks, "->", e)

    print("CP latent ranks:", list(cp_latent.keys()))
    print("PCA latent ks:", list(pca_latent.keys()))
