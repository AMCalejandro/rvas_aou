import hail as hl
import pandas as pd
import numpy as np
import argparse
import hailtop.batch as hb
from numpy.linalg import LinAlgError
from scipy.special import gammaincc

from utils.processing import filter_gene_matrix, get_significant_genes, get_remaining_genes


# ---------------------------------------------------------------------------
# linear-algebra helpers
# ---------------------------------------------------------------------------
def _nearest_psd(A, eps=1e-12):
    """Project a symmetric matrix to the nearest PSD by clipping eigenvalues."""
    A = (A + A.T) / 2.0
    w, V = np.linalg.eigh(A)
    w = np.clip(w, a_min=eps, a_max=None)
    return (V * w) @ V.T

def _scaled_cov(v, P):
    """diag(v) @ P @ diag(v) without forming the diagonal matrices."""
    v = np.asarray(v, dtype=float)
    return (v[:, None] * P) * v[None, :]

def _chol_logdet(Sigma, abs_ridge=1e-10, rel_ridge=1e-6):
    """
    Robust Cholesky of a (near-)SPD matrix.

    Returns (L, logdet, event, rel_jitter) where event is one of:
        'none'       -> plain Cholesky succeeded, no regularization
        'jitter_abs' -> jitter added and abs_ridge was the binding term
        'jitter_rel' -> jitter added and rel_ridge * mean_diag was binding
        'psd'        -> jitter insufficient; nearest-PSD projection applied
    and rel_jitter = (jitter added) / (mean diagonal variance), i.e. the size of
    the correction relative to the data scale. rel_jitter << 1 is cosmetic;
    values approaching or exceeding ~1e-2 mean regularization is materially
    reshaping Sigma and eating signal.
    """
    k = Sigma.shape[0]
    Sigma = (Sigma + Sigma.T) / 2.0

    try:
        L = np.linalg.cholesky(Sigma)
        return L, 2.0 * np.sum(np.log(np.diag(L))), "none", 0.0
    except LinAlgError:
        pass

    diag = np.diag(Sigma)
    avg_var = np.mean(diag) if np.all(np.isfinite(diag)) else 1.0
    if not np.isfinite(avg_var) or avg_var <= 0:
        avg_var = 1.0
    rel_term = rel_ridge * avg_var
    jitter = max(abs_ridge, rel_term)
    binding = "jitter_abs" if abs_ridge >= rel_term else "jitter_rel"
    rel_jitter = jitter / avg_var

    try:
        L = np.linalg.cholesky(Sigma + jitter * np.eye(k))
        return L, 2.0 * np.sum(np.log(np.diag(L))), binding, rel_jitter
    except LinAlgError:
        Sigma = _nearest_psd(Sigma) + jitter * np.eye(k)
        L = np.linalg.cholesky(Sigma)
        return L, 2.0 * np.sum(np.log(np.diag(L))), "psd", rel_jitter

def _logpdf_quad_from_chol(diff, L, logdet):
    """Return (log-density, Mahalanobis quadratic form) for an MVN via Cholesky."""
    k = diff.shape[0]
    y = np.linalg.solve(L, diff)      # L y = diff
    quad = float(y @ y)
    logpdf = -0.5 * (k * np.log(2.0 * np.pi) + logdet + quad)
    return logpdf, quad

def _robust_maha(diff, w, V, rel_tol=1e-8):
    """
    Mahalanobis distance restricted to the well-supported eigen-subspace of Sigma.

    w, V come from eigh(Sigma). Eigen-directions with eigenvalue below
    rel_tol * max_eigenvalue are dropped (the data cannot constrain the fit
    there), which keeps the distance finite when Sigma is (near-)rank-deficient.
    Returns (quad, effective_rank). Use effective_rank as the chi-square dof.
    """
    wmax = w[-1]
    if not np.isfinite(wmax) or wmax <= 0:
        return np.nan, 0
    keep = w > rel_tol * wmax
    proj = V.T @ diff                 # coordinates in the eigenbasis
    quad = float(np.sum((proj[keep] ** 2) / w[keep]))
    return quad, int(np.count_nonzero(keep))

def _chi2_sf(x, dof):
    """Survival function of chi-square: P(X > x) for X ~ chi2(dof)."""
    if dof <= 0 or not np.isfinite(x):
        return np.nan
    return float(gammaincc(dof / 2.0, x / 2.0))


# ---------------------------------------------------------------------------
# ClinVar-derived class priors
# ---------------------------------------------------------------------------
def estimate_lof_concordance(p_class, p_floor, p_ceiling):
    """
    Convert a raw ClinVar pathogenicity rate for a variant class into a
    background-corrected "LoF concordance index" in [0, 1].
    
    p_floor and p_ceiling as that noise floor and achievable ceiling and linearly rescaling
    removes most of the shared bias.
        concordance = clip((p_class - p_floor) / (p_ceiling - p_floor), 0, 1)

    p_class:    (Likely_pathogenic + Pathogenic) / total for the class of
                interest (e.g. Missense).
    p_floor:    same quantity for a class with ~no true functional impact
                (Synonymous is the classic choice; average with "Other" if
                you want a slightly more conservative floor).
    p_ceiling:  same quantity for the reference "definitely disruptive" class
                (LoF here) -- NOT assumed to be 1.0.
    """
    if p_ceiling <= p_floor:
        raise ValueError("p_ceiling must exceed p_floor")
    return float(np.clip((p_class - p_floor) / (p_ceiling - p_floor), 0.0, 1.0))

def class_prior_weights(
    concordance,
    delta_values=(1.0, 0.0, -1.0, 0.5, -0.5),
    anti_floor=0.02,
    partial_share=0.35,
):
    """
    Expand a scalar LoF-concordance index (see estimate_lof_concordance) into
    a component_weights vector over `delta_values`, ready to pass into
    compute_scallion_scores.

    concordance:    prior probability mass assigned to the LoF-concordant side
                    of the mixture (delta > 0), in [0, 1].
    anti_floor:     minimum prior mass reserved for the anti-concordant side
                    (delta < 0). A pathogenicity rate alone can't distinguish
                    "not damaging" from "damaging via dominant-negative /
                    gain-of-function" (opposite direction to LoF), so this
                    keeps that side from being silently zeroed out. Raise it
                    for genes/gene sets with known GOF or dominant-negative
                    mechanisms.
    partial_share:  fraction of the concordant mass assigned to the +0.5
                    (partial LoF) component rather than +1 (full LoF); the
                    same split is mirrored on the anti side. 0 -> all mass on
                    the +-1 components, 1 -> all mass on the +-0.5 components.

    Returns an array aligned to `delta_values`, summing to 1.
    """
    concordance = float(np.clip(concordance, 0.0, 1.0))
    anti_floor = float(np.clip(anti_floor, 0.0, 1.0))
    concordant_mass = concordance * (1.0 - anti_floor)
    null_mass = 1.0 - concordant_mass - anti_floor

    lookup = {
        1.0: concordant_mass * (1.0 - partial_share),
        0.5: concordant_mass * partial_share,
        0.0: null_mass,
        -0.5: anti_floor * partial_share,
        -1.0: anti_floor * (1.0 - partial_share),
    }

    weights = np.empty(len(delta_values))
    for i, d in enumerate(delta_values):
        matches = [k for k in lookup if np.isclose(k, d)]
        if not matches:
            raise ValueError(
                f"class_prior_weights has no slot for delta={d}; only "
                f"{{1, 0.5, 0, -0.5, -1}} are supported by default. Extend "
                f"`lookup` above if you use custom delta_values."
            )
        weights[i] = lookup[matches[0]]
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# main scoring function
# ---------------------------------------------------------------------------
def compute_scallion_scores(
    gene: str,
    beta_lof: np.ndarray,                 # (T,)
    se_lof: np.ndarray,                   # (T,)
    P: np.ndarray,                        # (T, T) null trait-correlation
    missense_betas,                       # (N, T) DataFrame or ndarray
    missense_ses,                         # (N, T) or (T,)
    mask_missing: bool = True,
    min_traits: int = 3,
    delta_values=(+1.0, 0.0, -1.0, +0.5, -0.5),
    component_weights=False,               # prior over the 5 mixture components
                                           # (order = delta_values). None -> uniform.
                                           # Build one from ClinVar class rates via
                                           # class_prior_weights(estimate_lof_concordance(...)).
    include_lof_uncertainty: bool = True,
    abs_ridge: float = 1e-10,
    rel_ridge: float = 1e-6,
    fit_dof=None,                         # dof for fit_pvalue; default k (traits used)
) -> pd.DataFrame:

    beta_lof = np.asarray(beta_lof, dtype=float)
    se_lof = np.asarray(se_lof, dtype=float)
    P = np.asarray(P, dtype=float)
    delta_values = np.asarray(delta_values, dtype=float)
    n_comp = delta_values.shape[0]

    # grab ids BEFORE converting betas
    if hasattr(missense_betas, "index"):
        snp_id = missense_betas.index.to_numpy()
    else:
        snp_id = np.arange(np.shape(missense_betas)[0])
    missense_betas = np.asarray(missense_betas, dtype=float)

    missense_ses = np.asarray(missense_ses, dtype=float)
    if missense_ses.ndim == 1:
        missense_ses = np.broadcast_to(missense_ses[None, :], missense_betas.shape)

    N, T = missense_betas.shape
    assert beta_lof.shape == (T,)
    assert se_lof.shape == (T,)
    assert P.shape == (T, T)

    # --- mixture prior over components -------------------------------------
    if not component_weights:
        log_w = np.zeros(n_comp)          # uniform -> constant, cancels
    else:
        p_lof = 0.9438
        p_missense = 0.2965
        p_floor = 0.0890   # Synonymous

        concordance = estimate_lof_concordance(p_missense, p_floor, p_lof)
        print(f"Missense LoF concordance index (background-corrected): {concordance:.3f}")
        weights = class_prior_weights(concordance)
        print("component_weights (order = [+1, 0, -1, +0.5, -0.5]):")
        print(" ", weights)

        w = np.asarray(weights, dtype=float)
        assert w.shape == (n_comp,) and np.all(w > 0)
        log_w = np.log(w / w.sum())

    P = (P + P.T) / 2.0
    gene_P_repaired = False
    try:
        np.linalg.cholesky(P + 1e-12 * np.eye(T))
    except LinAlgError:
        P = _nearest_psd(P, eps=1e-8)
        gene_P_repaired = True

    delta2 = delta_values ** 2
    uniq_d2 = np.unique(delta2)

    def _find(d):
        hits = np.where(np.isclose(delta_values, d))[0]
        return int(hits[0]) if hits.size else None

    i_lof, i_null, i_anti = _find(+1.0), _find(0.0), _find(-1.0)
    i_plof, i_panti = _find(+0.5), _find(-0.5)

    out = {
        "markerID": np.full(N, None, dtype=object),
        "gene_symbol": np.full(N, gene, dtype=object),
        "max_logpdf_raw": np.full(N, np.nan),
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
        # --- QC / diagnostics ---
        "min_mahalanobis": np.full(N, np.nan),
        "fit_pvalue": np.full(N, np.nan),
        "effective_rank": np.zeros(N, dtype=int),
        "reg_abs_fired": np.zeros(N, dtype=int),
        "reg_rel_fired": np.zeros(N, dtype=int),
        "reg_psd_fired": np.zeros(N, dtype=int),
        "regularized": np.zeros(N, dtype=bool),
        "max_rel_jitter": np.zeros(N, dtype=float),
        "gene_P_repaired": np.full(N, gene_P_repaired, dtype=bool),
    }

    for i in range(N):
        out["markerID"][i] = snp_id[i]    # traceable even if skipped

        b = missense_betas[i, :]
        se = missense_ses[i, :]

        ok = (
            np.isfinite(b) & np.isfinite(se)
            & np.isfinite(beta_lof) & np.isfinite(se_lof)
            & (se > 0) & (se_lof > 0)
        )

        if mask_missing:
            idx = np.where(ok)[0]
        else:
            if not np.all(ok):
                continue
            idx = np.arange(T)

        k = idx.size
        if k < min_traits:
            continue

        b_i = b[idx]
        lof_i = beta_lof[idx]
        se_i = se[idx]
        se_lof_i = se_lof[idx]
        P_i = P[np.ix_(idx, idx)]

        # covariance factored once per distinct delta^2; track regularization
        S_miss = _scaled_cov(se_i, P_i)
        chol_cache = {}
        eig_cache = {}
        reg_events = []
        rel_jitters = []
        if include_lof_uncertainty:
            S_lof = _scaled_cov(se_lof_i, P_i)
            for d2 in uniq_d2:
                Sig = S_miss + d2 * S_lof
                L, logdet, ev, rj = _chol_logdet(Sig, abs_ridge, rel_ridge)
                chol_cache[d2] = (L, logdet)
                eig_cache[d2] = np.linalg.eigh((Sig + Sig.T) / 2.0)  # (w, V)
                reg_events.append(ev)
                rel_jitters.append(rj)
        else:
            L, logdet, ev, rj = _chol_logdet(S_miss, abs_ridge, rel_ridge)
            we, Ve = np.linalg.eigh((S_miss + S_miss.T) / 2.0)
            chol_cache = {d2: (L, logdet) for d2 in uniq_d2}
            eig_cache = {d2: (we, Ve) for d2 in uniq_d2}
            reg_events.append(ev)
            rel_jitters.append(rj)

        out["reg_abs_fired"][i] = sum(e == "jitter_abs" for e in reg_events)
        out["reg_rel_fired"][i] = sum(e == "jitter_rel" for e in reg_events)
        out["reg_psd_fired"][i] = sum(e == "psd" for e in reg_events)
        out["regularized"][i] = any(e != "none" for e in reg_events)
        out["max_rel_jitter"][i] = max(rel_jitters) if rel_jitters else 0.0

        # per-component log-density (Cholesky) and robust Mahalanobis (eigen)
        raw_lps = np.empty(n_comp)
        robust_quads = np.empty(n_comp)
        eff_ranks = np.empty(n_comp, dtype=int)
        for j in range(n_comp):
            L, logdet = chol_cache[delta2[j]]
            diff = b_i - delta_values[j] * lof_i
            raw_lps[j], _ = _logpdf_quad_from_chol(diff, L, logdet)
            we, Ve = eig_cache[delta2[j]]
            robust_quads[j], eff_ranks[j] = _robust_maha(diff, we, Ve)

        # goodness-of-fit: closest LoF-concordant hypothesis, on the
        # well-supported subspace (dof = effective rank of that component)
        jmin = int(np.argmin(robust_quads))
        d2_min = float(robust_quads[jmin])
        eff_dof = eff_ranks[jmin] if fit_dof is None else int(fit_dof)
        out["min_mahalanobis"][i] = d2_min
        out["effective_rank"][i] = eff_ranks[jmin]
        out["fit_pvalue"][i] = _chi2_sf(d2_min, eff_dof)

        # posterior over components (prior + likelihood)
        log_post = raw_lps + log_w
        shifted = log_post - np.max(log_post)
        probs = np.exp(shifted)
        probs_sum = probs.sum()
        if probs_sum == 0 or not np.isfinite(probs_sum):
            continue
        probs /= probs_sum

        scallion_i = float(np.sum(probs * delta_values))
        var_scallion_i = float(np.sum(probs * (delta_values - scallion_i) ** 2))

        denom = np.linalg.norm(b_i) * np.linalg.norm(lof_i)
        align = float((b_i @ lof_i) / denom) if denom > 0 else np.nan

        out["max_logpdf_raw"][i] = float(np.max(raw_lps))
        if i_lof is not None:
            out["logpdf_lof"][i] = raw_lps[i_lof]
            out["scallion_prob_lof"][i] = probs[i_lof]
        if i_null is not None:
            out["logpdf_null"][i] = raw_lps[i_null]
        if i_anti is not None:
            out["logpdf_anti"][i] = raw_lps[i_anti]
        if i_plof is not None:
            out["logpdf_partial_lof"][i] = raw_lps[i_plof]
        if i_panti is not None:
            out["logpdf_partial_anti"][i] = raw_lps[i_panti]
        if i_lof is not None and i_null is not None:
            out["scallion_llr"][i] = raw_lps[i_lof] - raw_lps[i_null]
        if i_lof is not None and i_anti is not None:
            out["scallion_prob_lof_signed"][i] = probs[i_lof] - probs[i_anti]

        out["scallion_prob_mixture"][i] = scallion_i
        out["scallion_prob_mixture_var"][i] = var_scallion_i
        out["alignment"][i] = align
        out["traits_used"][i] = k

    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# processing helpers
# ---------------------------------------------------------------------------
def get_gene_phenotype_correlations(gene_ht_path, phenos_cor_path: str, gene: str):
    """Get phenotype correlations for a given gene."""
    
    phenos_cor = hl.read_table(phenos_cor_path)
    ht = hl.read_table(gene_ht_path)

    # Code below is to work on the single phenotype case
    # phenos_cor = phenos_cor.filter(
    #     (phenos_cor.i_pheno == "2453") & (phenos_cor.i_coding == "") &
    #     (phenos_cor.j_pheno == "2453") & (phenos_cor.j_coding == "") 
    #     )
    # phenos_cor = phenos_cor.filter(
    #     (phenos_cor.i_pheno == "20002") & (phenos_cor.i_coding == "1473") &
    #     (phenos_cor.j_pheno == "20002") & (phenos_cor.j_coding == "1473") 
    # )

    b_plof = ht.filter(ht.gene_symbol == gene).key_by().select(
        "phenocode", "coding", "BETA_Burden"
    )

    pheno_coding_pairs = b_plof.select('phenocode', 'coding').collect()
    pheno_coding_set = {(row.phenocode, row.coding) for row in pheno_coding_pairs}
    phenos_cor = phenos_cor.filter(
        hl.literal(pheno_coding_set).contains((phenos_cor.i_pheno, phenos_cor.i_coding)) &
        hl.literal(pheno_coding_set).contains((phenos_cor.j_pheno, phenos_cor.j_coding))
    )
    
    phenos_cor_pairs_set = phenos_cor.aggregate(
        hl.agg.collect_as_set(
            hl.tuple([phenos_cor.i_pheno, phenos_cor.i_coding])
        ).union(
            hl.agg.collect_as_set(
                hl.tuple([phenos_cor.j_pheno, phenos_cor.j_coding])
            )
        )
    )
    
    missing_pheno_coding_pairs = pheno_coding_set - phenos_cor_pairs_set
    
    if missing_pheno_coding_pairs:
        print(f'There are missing phenotypes in the correlation matrix for {gene}')
        b_plof = b_plof.filter(
            ~hl.literal(missing_pheno_coding_pairs).contains(
                hl.tuple([b_plof.phenocode, b_plof.coding])
            )
        )
        
    pheno_coding_keys = b_plof.aggregate(
        hl.agg.collect_as_set(
            hl.struct(phenocode=b_plof.phenocode, coding=b_plof.coding)
        )
    )
    
    blof_pd = b_plof.to_pandas()
    blof_pd['key'] = blof_pd["phenocode"] + "_" + blof_pd["coding"].fillna("")

    pheno_corr_pd = phenos_cor.to_pandas()
    pheno_corr_pd["i_key"] = pheno_corr_pd["i_pheno"] + "_" + pheno_corr_pd["i_coding"].fillna("")
    pheno_corr_pd["j_key"] = pheno_corr_pd["j_pheno"] + "_" + pheno_corr_pd["j_coding"].fillna("")
    pivot_matrix = pheno_corr_pd.pivot_table(
        index="i_key", columns="j_key", values="entry", fill_value=0
    )
    
    return pheno_coding_keys, blof_pd, pivot_matrix

def filter_variant_matrix(var_path: str, gene: str, pheno_coding_keys, save_path: str):
    """Filter variant-level matrix for given gene and pheno_coding_keys."""
    var_mt = hl.read_matrix_table(var_path)
    
    var_mt_filtered = hl.filter_intervals(
        var_mt, 
        hl.experimental.get_gene_intervals(gene_symbols=[gene], reference_genome='GRCh38') 
    )
    
    var_mt_filtered = var_mt_filtered.filter_cols(
        hl.literal(pheno_coding_keys).contains(
            hl.struct(phenocode=var_mt_filtered.phenocode, coding=var_mt_filtered.coding)
        )
    )

    missense_annots = ["missense"]
    plof_annots = ["pLoF"]
    var_mt_filtered = var_mt_filtered.filter_rows(
        (var_mt_filtered.gene == gene)
        & (hl.literal(missense_annots + plof_annots).contains(var_mt_filtered.annotation))
    )
    var_mt_filtered_ht = var_mt_filtered.entries()
    var_mt_filtered_ht = var_mt_filtered_ht.naive_coalesce(10)
    var_mt_filtered_ht = var_mt_filtered_ht.checkpoint(save_path, overwrite=True)
    var_mt_filtered_ht = var_mt_filtered_ht.key_by()
    
    var_missense = var_mt_filtered_ht.filter(
        hl.literal(missense_annots).contains(var_mt_filtered_ht.annotation)
    )

    var_missense = var_missense.select("markerID", "phenocode", "coding", "BETA", "SE", "AC", "AF")
    #var_missense.export(f"gs://aou_amc/tmp/scallion_v2/var_missense_{gene}.csv")
    var_missense = var_missense.to_pandas()#.drop(columns=["locus", "alleles"])
    var_missense['key'] = var_missense["phenocode"] + "_" + var_missense["coding"].fillna("")

    var_plof_entries = var_mt_filtered_ht.filter(
        hl.literal(plof_annots).contains(var_mt_filtered_ht.annotation)
    )
    
    var_plof = var_plof_entries.group_by(
        var_plof_entries.phenocode, 
        var_plof_entries.coding
    ).aggregate(
        # Inverse variance weights: w_i = 1/SE_i^2
        # Weighted beta: sum(BETA_i * w_i) / sum(w_i)
        # Weighted SE: sqrt(1 / sum(w_i))
        sum_weights = hl.agg.sum(1.0 / (var_plof_entries.SE ** 2)),
        weighted_beta_sum = hl.agg.sum(var_plof_entries.BETA / (var_plof_entries.SE ** 2)),
        n_variants = hl.agg.count()
    )
    
    var_plof = var_plof.annotate(
        BETA_meta = var_plof.weighted_beta_sum / var_plof.sum_weights,
        SE_meta = hl.sqrt(1.0 / var_plof.sum_weights)
    )

    var_plof = var_plof.key_by()
    var_plof = var_plof.select(
        "phenocode", "coding", "BETA_meta", "SE_meta", "n_variants"
    )
    #var_plof.export(f"gs://aou_amc/tmp/scallion_v2/var_plof_{gene}.csv")    
    var_plof = var_plof.to_pandas()
    var_plof['key'] = var_plof["phenocode"] + "_" + var_plof["coding"].fillna("")

    return var_mt_filtered_ht, var_missense, var_plof


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def process_gene(
    gene: str,
    gene_ht_path,
    phenos_cor_path: str,
    var_path: str,
    results_prefix: str,
    permute: bool = True,
    permute_seed: int = None,
):
    """Run full pipeline for a single gene.

    Parameters
    ----------
    permute : bool
        If True, randomly shuffle the LoF beta vector (var_plof_pd['BETA_meta'])
        before scoring. Useful as a null/permutation control. A length-1 vector
        is unaffected by shuffling.
    permute_seed : int, optional
        Seed for the permutation, for reproducibility. If None, uses global RNG state.
    """
    hl.init(
        master='local[4]',
        tmp_dir='gs://aou_tmp',
        gcs_requester_pays_configuration='aou-neale-gwas',
        worker_memory="12g",
        worker_cores=4,
        default_reference="GRCh38",
    )
    
    pheno_coding_keys, blof_pd, pivot_matrix = get_gene_phenotype_correlations(
        gene_ht_path=gene_ht_path,
        phenos_cor_path=phenos_cor_path,
        gene=gene,
    )
    blof_pd = blof_pd.sort_values("key")

    save_path = f"{results_prefix}/tmp/{gene}_variants.mt"
    _, var_missense_pd, var_plof_pd = filter_variant_matrix(
        var_path, gene, pheno_coding_keys, save_path
    )

    var_missense_pd = var_missense_pd.sort_values("key")
    var_plof_pd = var_plof_pd.sort_values("key")

    assert (
        blof_pd.shape[0]
        == len(var_missense_pd.key.unique())
        == len(var_plof_pd.key.unique())
        == pivot_matrix.shape[0]
    ), "Row counts do not match across datasets."

    assert (
        list(var_plof_pd.key) == list(var_missense_pd.key.unique()) == list(pivot_matrix.index)
    ), "Phenocode order mismatch between plof, missense, or pivot_matrix."

    ac_mean_missense = (
        var_missense_pd
        .groupby('markerID', as_index=False)['AC']
        .mean()
        .rename(columns={'AC': 'mean_AC'})
    )
    missense_beta_pd = (
        var_missense_pd
        .pivot(index='markerID', columns='key', values='BETA')
        .astype(float)
    )
    missense_se_pd = (
        var_missense_pd
        .pivot(index='markerID', columns='key', values='SE')
        .astype(float)
    )

    beta_lof = var_plof_pd['BETA_meta'].values
    se_lof = var_plof_pd['SE_meta'].values
    
    if permute:
        print('Running permutation')
        rng = np.random.default_rng(permute_seed)
        idx = rng.permutation(len(beta_lof))
        beta_lof = beta_lof[idx]
        se_lof = se_lof[idx]
    
    df_scores = compute_scallion_scores(
        gene=gene,
        beta_lof=beta_lof,
        se_lof=se_lof,
        P=pivot_matrix,
        missense_betas=missense_beta_pd,
        missense_ses=missense_se_pd,
        mask_missing=True,
        min_traits=1,
    )

    df_scores = df_scores.merge(ac_mean_missense, on='markerID', how='left')
    print(df_scores)
    out_path = f"{results_prefix}/{gene}.csv"
    df_scores_ht = hl.Table.from_pandas(df_scores)
    try:
        df_scores_ht.export(out_path)
    except:
        out_path = f"{results_prefix}/{gene}_retry.ht"
        df_scores_ht.export(out_path)
    
    print(f"[✔] Saved results for {gene} → {out_path}")




def main(args):
    hl.init(
        app_name='Running scallion',
        idempotent=True,
        tmp_dir='gs://aou_tmp',
        default_reference="GRCh38",
        gcs_requester_pays_configuration="aou-neale-gwas",
        log="/run_scallion.log",
    )

    genebass_gene_path = "gs://ukbb-exome-public/500k/results/results.mt"
    phenos_cor_path    = "gs://ukbb-exome-public/500k/qc/correlation_table_phenos_500k.ht"
    var_path           = "gs://ukbb-exome-public/500k/results/variant_results.mt"
    results_prefix     = "gs://aou_amc/scallion/results_final"
    save_out_ht        = 'gs://aou_amc/scallion/dev/pLoF_genebass_significant_nosparse.ht'

    # --- Load or compute significant genes ---
    if not hl.hadoop_exists(save_out_ht):
        gene_mt_clean = filter_gene_matrix(genebass_gene_path, phenos_cor_path, pval_threshold=2.5e-6)
        _, plof_genes = get_significant_genes(gene_mt_clean, save_out_ht, mode=args.scallion_mode)

    if not args.run_scallion:
        return
    # --- Override gene list from CLI if provided ---
    if args.genes:
        plof_genes = args.genes
        print(f"Using manually specified genes: {plof_genes}")
    elif args.search_best_scallion:
        path_data = 'gs://aou_amc/scallion/data/experiments/clinvar_groundtruth/high_deleterious_missense.csv'
        genes_high = pd.read_csv(path_data, sep = '\t').gene.tolist()
        path_data = 'gs://aou_amc/scallion/data/experiments/clinvar_groundtruth/low_deleterious_missense.csv'
        genes_low = pd.read_csv(path_data, sep = '\t').gene.tolist()
        plof_genes = genes_high + genes_low
    else:
        genes_lof_path = ".".join(save_out_ht.rsplit(".", 1)[:-1]) + f"_{args.scallion_mode}.txt"
        plof_genes = pd.read_csv(genes_lof_path, header=None)[0].tolist()
        print(f"Loaded {len(plof_genes)} genes from cache.")

    # --- Test mode ---
    if args.test:
        plof_genes = plof_genes[:3]
        print(f"TEST MODE: {len(plof_genes)} genes: {plof_genes}")
    else:
        plof_genes = get_remaining_genes(plof_genes, results_prefix, args.overwrite)

    print(f"Processing {len(plof_genes)} genes.")
    if not plof_genes:
        print("Nothing to do.")
        return

    # --- Submit Hail Batch jobs ---
    backend = hb.ServiceBackend(
        billing_project="all-by-aou",
        remote_tmpdir="gs://aou_tmp/v8/",
    )
    b = hb.Batch(
        name="scallion run it",
        requester_pays_project="aou-neale-gwas",
        backend=backend,
    )

    for gene in plof_genes:
        j = b.new_python_job(f'run_scallion_{gene}', attributes={"gene": gene})
        j.image("hailgenetics/hail:0.2.133-py3.11")
        j.memory('16Gi')   # container memory must exceed driver memory below
        j.env('PYSPARK_SUBMIT_ARGS', '--driver-memory 12g --executor-memory 12g pyspark-shell')
        j.call(process_gene, gene, save_out_ht, phenos_cor_path, var_path, results_prefix)

    b.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run scallion.')
    
    parser.add_argument('--run-scallion',
                        action='store_true', 
                        help='Run scallion')
    
    parser.add_argument('--overwrite',
                        help='Overwrite scallion results', 
                        action='store_true'
    )

    parser.add_argument('--genes',
                    type=lambda s: [g.strip() for g in s.split(',') if g.strip()],
                    default=None,
                    help='Comma-separated list of genes to run (overrides other gene sources)')
    
    parser.add_argument('--test', 
                        action='store_true', 
                        help='Use first 3 genes for testing')
    
    parser.add_argument('--search-best-scallion', 
                        action='store_true', 
                        help='Use first 3 genes for testing')
    
    parser.add_argument('--scallion-mode',
                    type=str,
                    choices=['multi', 'single'],
                    help='Use first 3 genes for testing')
    
    args = parser.parse_args()
    main(args)
