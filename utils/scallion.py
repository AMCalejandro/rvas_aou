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
    se_lof: np.ndarray,                # shape (T,)
    P: np.ndarray,                     # shape (T,T), correlation under null
    missense_betas: np.ndarray,        # shape (N,T)
    missense_ses: np.ndarray,          # shape (N,T) or (T,)
    mask_missing: bool = True,
    min_traits: int = 3,
    ridge: float = 1e-6,
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
    se_lof = np.asarray(se_lof, dtype=float)
    P = np.asarray(P, dtype=float)

    snp_id = missense_betas.index.to_numpy()

    if missense_ses.ndim == 1:
        missense_ses = np.broadcast_to(missense_ses[None, :], missense_betas.shape)
    else:
        missense_ses = np.asarray(missense_ses, dtype=float)
        missense_betas = np.asarray(missense_betas, dtype=float)

    N, T = missense_betas.shape
    assert beta_lof.shape == (T,)
    assert se_lof.shape == (T,)
    assert P.shape == (T, T)

    # Symmetrize P and ensure PSD
    P = (P + P.T) / 2.0
    try:
        np.linalg.cholesky(P + 1e-12 * np.eye(T))
    except LinAlgError:
        P = _nearest_psd(P, eps=1e-8)

    # mixture components
    delta_values = np.array([+1, 0, -1, +0.5, -0.5]) # Lof, Null, Anti, Partial Lof, Partial Anti

    out = {
        "markerID": np.full(N, None, dtype=object),
        "gene_symbol": np.full(N, None, dtype=object),
        "logpdf_lof": np.full(N, np.nan),
        "logpdf_partial_lof": np.full(N, np.nan),
        "logpdf_null": np.full(N, np.nan),
        "logpdf_partial_anti": np.full(N, np.nan),
        "logpdf_anti": np.full(N, np.nan),
        "scallion_llr": np.full(N, np.nan),
        "scallion_prob_lof": np.full(N, np.nan),
        "scallion_prob_lof_signed": np.full(N, np.nan),
        "scallion_prob_mixture": np.full(N, np.nan),
        "scallion_prob_mixture_var": np.full(N, np.nan),
        "alignment": np.full(N, np.nan),
        "traits_used": np.zeros(N, dtype=int),
    }

    for i in range(N):
        b = missense_betas[i, :]
        se = missense_ses[i, :]

        ok = (
            np.isfinite(b)
            & np.isfinite(se)
            & np.isfinite(beta_lof)
            & np.isfinite(se_lof)
            & (se > 0)
            & (se_lof > 0)
        )

        if mask_missing:
            idx = np.where(ok)[0]
        else:
            if not np.all(ok):
                continue
            idx = np.arange(T)

        k = len(idx)
        if k < min_traits:
            continue

        b_i = b[idx]
        lof_i = beta_lof[idx]
        se_i = se[idx]
        se_lof_i = se_lof[idx]
        P_i = P[np.ix_(idx, idx)]

        # Ensure P_i is PSD
        if np.any(np.linalg.eigvalsh(P_i) <= 0):
            P_i += np.eye(k) * ridge

        # Build Sigma = V_total P_i V_total
        se_total = np.sqrt(se_i**2 + se_lof_i**2)
        V_total = np.diag(se_total)
        Sigma = V_total @ P_i @ V_total

        # Regularize Sigma if needed
        try:
            np.linalg.cholesky(Sigma)
        except LinAlgError:
            Sigma += np.eye(k) * ridge

        # Define mixture means dynamically
        means = [d * lof_i for d in delta_values]

        # Compute log probabilities for each component
        lps = np.array([
            _logpdf_mvn(b_i, mean=m, Sigma=Sigma)
            for m in means
        ])

        lps -= np.max(lps)
        probs = np.exp(lps)
        probs_sum = np.sum(probs)
        if probs_sum == 0 or not np.isfinite(probs_sum):
            continue
        probs /= probs_sum

        # Posterior mixture mean/variance
        scallion_i = np.sum(probs * delta_values)
        var_scallion_i = np.sum(probs * (delta_values - scallion_i) ** 2)

        #` Alignme`nt
        denom = np.linalg.norm(b_i) * np.linalg.norm(lof_i)
        align = (b_i @ lof_i) / denom if denom > 0 else np.nan

        # Assign outputs
        out["markerID"][i] = snp_id[i]
        out["gene_symbol"][i] = gene
        out["logpdf_lof"][i] = lps[0]
        out["logpdf_partial_lof"][i] = lps[3]
        out["logpdf_null"][i] = lps[1]
        out["logpdf_partial_anti"][i] = lps[4]
        out["logpdf_anti"][i] = lps[2]
        out["scallion_llr"][i] = lps[0] - lps[1]
        out["scallion_prob_lof"][i] = probs[0]
        out["scallion_prob_lof_signed"][i] = probs[0] - probs[2]
        out["scallion_prob_mixture"][i] = scallion_i
        out["scallion_prob_mixture_var"][i] = var_scallion_i
        out["alignment"][i] = align
        out["traits_used"][i] = k

    return pd.DataFrame(out)