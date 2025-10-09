import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError


def _nearest_psd(A, eps=1e-10):
    """Project a symmetric matrix to the nearest PSD by clipping eigenvalues."""
    A = (A + A.T) / 2.0
    w, V = np.linalg.eigh(A)
    w_clipped = np.clip(w, a_min=eps, a_max=None)
    return (V * w_clipped) @ V.T

def _logpdf_mvn(x, mean, Sigma):
    """
    Numerically stable log-density of MVN using Cholesky.
    x, mean: (k,)
    Sigma: (k,k), assumed SPD (or very nearly)
    """
    k = x.shape[0]
    try:
        L = np.linalg.cholesky(Sigma)
    except LinAlgError:
        # Small diagonal jitter then try again
        jitter = 1e-8 * np.mean(np.diag(Sigma))
        if not np.isfinite(jitter) or jitter == 0:
            jitter = 1e-8
        Sigma = Sigma + jitter * np.eye(k)
        L = np.linalg.cholesky(_nearest_psd(Sigma))
    
    diff = x - mean
    # Solve L * y = diff
    y = np.linalg.solve(L, diff)
    quad = y @ y
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (k * np.log(2.0 * np.pi) + logdet + quad)

def compute_scallion_scores(
    gene: str,
    beta_lof: np.ndarray,              # shape (T,)
    P: np.ndarray,                     # shape (T,T), correlation under null
    missense_betas: np.ndarray,        # shape (N,T)
    missense_ses: np.ndarray,          # shape (N,T) or (T,)
    mask_missing: bool = True,
    min_traits: int = 3
) -> pd.DataFrame:
    """
    Compute scallion for N missense variants across T traits.

    - beta_lof: effect vector for pLoF across traits
    - P: trait correlation matrix (symmetric, ~PSD)
    - missense_betas: N x T estimated effects for each missense variant
    - missense_ses: N x T SEs (or a single T-vector applied to all variants)
    - mask_missing: mask to traits where all needed pieces are finite
    - min_traits: require at least this many traits to score; else NaNs
    """
    beta_lof = np.asarray(beta_lof, dtype=float)
    # beta_lof = beta_lof.mean(axis=0)
    P = np.asarray(P, dtype=float)

    snp_id = missense_betas.index.to_numpy()

    if missense_ses.ndim == 1:
        missense_ses = np.broadcast_to(missense_ses[None, :], missense_betas.shape)
    else:
        missense_ses = np.asarray(missense_ses, dtype=float)
        missense_betas = np.asarray(missense_betas, dtype=float)

    N, T = missense_betas.shape
    assert beta_lof.shape == (T,)
    assert P.shape == (T, T)

    # Symmetrize P and push to PSD if needed
    P = (P + P.T) / 2.0
    try:
        # quick PD check via Cholesky; if fail, repair
        np.linalg.cholesky(P + 1e-12 * np.eye(T))
    except LinAlgError:
        P = _nearest_psd(P, eps=1e-8)

    out = {
        "markerID": np.full(N, None, dtype=object),
        "gene_symbol": np.full(N, None, dtype=object),
        "logpdf_lof": np.full(N, np.nan),
        "logpdf_null": np.full(N, np.nan),
        "logpdf_anti": np.full(N, np.nan),
        "scallion_llr": np.full(N, np.nan),          # lof vs null
        "scallion_prob_lof": np.full(N, np.nan),     # softmax over {lof, null, anti}
        "alignment": np.full(N, np.nan),             # cosine similarity for intuition
        "traits_used": np.zeros(N, dtype=int)
    }
    for i in range(N):
        b = missense_betas[i, :]
        se = missense_ses[i, :]

        ok = np.isfinite(b) & np.isfinite(se) & np.isfinite(beta_lof) & (se > 0) 
        if mask_missing:
            idx = np.where(ok)[0]
        else:
            if not np.all(ok):
                # if not masking, bail out with NaNs
                continue
            idx = np.arange(T)

        k = len(idx)
        if k < min_traits:
            continue

        b_i = b[idx]
        lof_i = beta_lof[idx]
        se_i = se[idx]
        P_i = P[np.ix_(idx, idx)]

        # Build Sigma = V P V
        V = np.diag(se_i)
        Sigma = V @ P_i @ V

        # Log-densities
        lp_lof  = _logpdf_mvn(b_i, mean=lof_i, Sigma=Sigma)
        lp_null = _logpdf_mvn(b_i, mean=np.zeros(k), Sigma=Sigma)
        lp_anti = _logpdf_mvn(b_i, mean=-lof_i, Sigma=Sigma)

        # LLR and softmax probability for LoF-direction model
        lps = np.array([lp_lof, lp_null, lp_anti])
        lps_center = lps - np.max(lps)  # numerical stability
        probs = np.exp(lps_center) / np.sum(np.exp(lps_center))

        # Alignment (cosine) just for signal direction intuition
        denom = (np.linalg.norm(b_i) * np.linalg.norm(lof_i))
        align = (b_i @ lof_i) / denom if denom > 0 else np.nan

        out["markerID"][i] = snp_id[i]
        out["gene_symbol"][i] = gene
        out["logpdf_lof"][i] = lp_lof
        out["logpdf_null"][i] = lp_null
        out["logpdf_anti"][i] = lp_anti
        out["scallion_llr"][i] = lp_lof - lp_null
        out["scallion_prob_lof"][i] = probs[0]
        out["alignment"][i] = align
        out["traits_used"][i] = k

    return pd.DataFrame(out)