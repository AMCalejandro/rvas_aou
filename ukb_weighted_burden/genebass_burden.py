import hail as hl
import argparse
import re
import math
from typing import List

import pandas as pd


# ── FlexRV weight grid constants ──────────────────────────────────────────────
CUBIC_ROOT_TRANSITIONS: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BETA_B_SCORE:           List[float] = [1.5, 2.0, 3.0, 5.0, 10.0, 20.0]
BETA_B_MAF:             List[float] = [2.0, 5.0]
MAC_THRESHOLDS:         List[int]   = [1, 3, 5, 10, 25, 50, 100, 200, 500]
SCORE_WEIGHT_KEYS: List[str] = (
    ["unweighted", "lof_only"]
    + [f"cubic_root_t{t}" for t in CUBIC_ROOT_TRANSITIONS]
    + [f"beta_b{b}"        for b in BETA_B_SCORE]
)
MAF_WEIGHT_KEYS: List[str] = (
    ["maf_unweighted"]
    + [f"maf_beta_b{b}" for b in BETA_B_MAF]
    + [f"maf_mac{mac}"  for mac in MAC_THRESHOLDS]
)
COMBINED_WEIGHT_KEYS: List[str] = [
    f"{sk}__{mk}" for sk in SCORE_WEIGHT_KEYS for mk in MAF_WEIGHT_KEYS
]
N_WEIGHTS: int = len(COMBINED_WEIGHT_KEYS)  # 16 × 12 = 192

def get_flexrv_weight_keys(score_fields: List[str]) -> List[str]:
    """Full ordered list of weight labels for a given set of score fields."""
    return [f"{sf}__{ck}" for sf in score_fields for ck in COMBINED_WEIGHT_KEYS]


# ── FlexRV primitives ──────────────────────────────────────────────
def _hl_cubic_root_weight(s, t):
    t_cbrt = t ** (1.0 / 3.0)
    denom  = abs(1.0 - t) ** (1.0 / 3.0) + t_cbrt
    numer  = hl.sign(s - t) * hl.abs(s - t) ** (1.0 / 3.0) + t_cbrt
    return numer / denom

def _hl_beta_score_weight(s, b):
    return hl.abs(s) ** (b - 1.0)

def _hl_maf_beta_weight(maf, b, maf_scale=0.001):
    return _hl_beta_score_weight(maf / maf_scale, b)

def _hl_maf_threshold_weight(
    maf:       hl.expr.Float64Expression,
    mac:       int,
    n_samples: hl.expr.NumericExpression,   # Hail expression — varies per phenotype
) -> hl.expr.Float64Expression:
    threshold = hl.float64(mac) / (2.0 * hl.float64(n_samples))
    return hl.if_else(maf <= threshold, hl.float64(1.0), hl.float64(0.0))

def _combined_weight_array_expr(
    s_expr:      hl.expr.Float64Expression,
    is_lof_expr: hl.expr.BooleanExpression,
    maf_expr:    hl.expr.Float64Expression,
    n_samples:   hl.expr.NumericExpression,  # Hail expression
) -> hl.expr.ArrayExpression:
    score_w = [
        hl.float64(1.0),
        hl.if_else(is_lof_expr, hl.float64(1.0), hl.float64(0.0)),
        *[_hl_cubic_root_weight(s_expr, t) for t in CUBIC_ROOT_TRANSITIONS],
        *[_hl_beta_score_weight(s_expr, b)  for b in BETA_B_SCORE],
    ]
    maf_w = [
        hl.float64(1.0),
        *[_hl_maf_beta_weight(maf_expr, b)                   for b in BETA_B_MAF],
        *[_hl_maf_threshold_weight(maf_expr, mac, n_samples) for mac in MAC_THRESHOLDS],
    ]
    return hl.array([sw * mw for sw in score_w for mw in maf_w])


def _entry_weight_dict(
    mt:           hl.MatrixTable,
    score_fields: List[str],
    maf_field:    str,
    n_samples:    hl.expr.NumericExpression,
) -> hl.expr.StructExpression:
    """
    Returns a Hail struct where each field is a score_field name,
    and the value is the N_WEIGHTS-length weight array for that score.

    e.g. struct { score_a: [w0, w1, ..., w191], score_b: [...] }
    """
    is_lof   = hl.coalesce(mt.annotation == 'pLoF', False)
    maf_expr = hl.coalesce(hl.float64(mt[maf_field]), 0.0)

    per_score = {}
    for sf in score_fields:
        s_expr = hl.if_else(
            is_lof,
            hl.float64(1.0),
            hl.coalesce(hl.float64(mt[sf]), hl.float64(0.0)),
        )
        per_score[sf] = _combined_weight_array_expr(s_expr, is_lof, maf_expr, n_samples)

    return hl.struct(**per_score)


# ── Cauchy Combination Test helpers ───────────────────────────────────────────
# Below this p-value, tan((0.5 - p) * pi) loses precision (0.5 - p rounds to
# 0.5 in float64 once p is within ~1e-16 of it), since the ulp of 0.5 is
# ~1.1e-16. Use the small-angle approximation tan(pi/2 - x) ≈ 1/x (x = p*pi)
# instead, which is exact in the limit and avoids that cancellation.
CCT_SMALL_P_CUTOFF = 1e-16
# Above this Cauchy-statistic magnitude, 0.5 - atan(T)/pi cancels down to
# noise (atan(T)/pi is already within float64 epsilon of 0.5). Switch to the
# Cauchy tail asymptote 1/(pi*T), which is what 0.5 - atan(T)/pi converges to
# analytically anyway — this is the standard fix used in the reference ACAT
# implementation (Liu & Xie 2020) and in STAAR/SAIGE-GENE+.
CCT_LARGE_T_CUTOFF = 1e15


def add_cct_p_entry(
    gene_mt:      hl.MatrixTable,
    score_fields: List[str],
) -> hl.MatrixTable:
    """
    CCT across the 192-weight axis for each (gene, phenotype) entry, with
    equal weights (1/n_valid) over the weight combos that produced a defined
    p-value.

    Expects entry field:
        p_arr: struct{ score_field: array<float64>[N_WEIGHTS] }

    Adds entry field:
        cct_p: struct{ score_field: float64 }
            One CCT p-value per gene × phenotype × score field.
    """
    pi = hl.float64(math.pi)

    def cauchy_term(p):
        p_clip = hl.max(hl.float64(0.0), hl.min(hl.float64(1.0), p))
        return hl.if_else(
            p_clip < CCT_SMALL_P_CUTOFF,
            hl.float64(1.0) / (p_clip * pi),
            hl.tan((hl.float64(0.5) - p_clip) * pi),
        )

    final = {}
    for sf in score_fields:
        p_arr = gene_mt.p_arr[sf]   # array<float64>[N_WEIGHTS]

        cauchy_terms = p_arr.map(
            lambda p: hl.if_else(hl.is_defined(p), cauchy_term(p), hl.missing(hl.tfloat64))
        )

        valid_terms = cauchy_terms.filter(hl.is_defined)
        n_valid     = hl.len(valid_terms)
        t_stat      = hl.sum(valid_terms) / hl.float64(n_valid)

        final[sf] = hl.if_else(
            n_valid > 0,
            hl.if_else(
                t_stat > CCT_LARGE_T_CUTOFF,
                hl.float64(1.0) / (t_stat * pi),
                hl.float64(0.5) - hl.atan(t_stat) / pi,
            ),
            hl.missing(hl.tfloat64),
        )

    return gene_mt.annotate_entries(
        cct_p=hl.struct(**final)
    )


def _list_flexrv_batch_paths(trait_type: str) -> List[str]:
    """
    Discovers the batch checkpoints run_flexrv_burden_bin/quant wrote for
    `trait_type` under FLEXRV_BASE_PATH/burden_results/ (one per --run_burden
    --burden_mode flexrv phenotype batch), in batch order.
    """
    out_prefix = FLEXRV_BATCH_OUT_PREFIX[trait_type]
    batch_dir  = f'{FLEXRV_BASE_PATH}burden_results'
    pattern    = re.compile(rf'{re.escape(out_prefix)}_batch(\d+)\.mt/?$')

    matches = []
    for entry in hl.hadoop_ls(batch_dir):
        m = pattern.search(entry['path'])
        if m:
            matches.append((int(m.group(1)), entry['path']))

    if not matches:
        raise FileNotFoundError(
            f"No FlexRV batch checkpoints found under {batch_dir} matching "
            f"'{out_prefix}_batch*.mt' — run --run_burden --burden_mode flexrv "
            f"--trait_type {trait_type} first."
        )
    return [path for _, path in sorted(matches)]


def combine_flexrv_cct(
    trait_type:    str,
    score_fields:  List[str] = None,
    sig_threshold: float     = None,
) -> hl.MatrixTable:
    """
    Merges the batched FlexRV burden checkpoints for `trait_type` (each batch
    holds a disjoint phenotype column slice over the same gene rows) and
    collapses the 192-weight grid into one CCT p-value per gene × phenotype ×
    score field via add_cct_p_entry.

    Writes:
      - the combined gene_mt (cct_p + n_var; the per-weight p_arr/z_arr are
        dropped once collapsed — the raw per-batch checkpoints they came from
        are untouched) ->
        FLEXRV_BASE_PATH/burden_results/{trait_type}_flexrv_cct.mt
      - a flattened TSV of nominally significant (cct_p < sig_threshold in
        any score field) gene × phenotype × score-field rows ->
        FLEXRV_BASE_PATH/{trait_type}_flexrv_cct.tsv
    """
    score_fields  = score_fields  or FLEXRV_PRIMARY_SCORE_FIELDS
    sig_threshold = sig_threshold if sig_threshold is not None else PHENO_SELECT_SIG_THRESHOLD

    batch_paths = _list_flexrv_batch_paths(trait_type)
    print(f"[flexrv_cct] Combining {len(batch_paths)} batch(es) for '{trait_type}': {batch_paths}")

    gene_mt = hl.read_matrix_table(batch_paths[0])
    for path in batch_paths[1:]:
        gene_mt = gene_mt.union_cols(hl.read_matrix_table(path))

    gene_mt = add_cct_p_entry(gene_mt, score_fields)
    gene_mt = gene_mt.drop('p_arr', 'z_arr')

    out_mt_path = f'{FLEXRV_BASE_PATH}burden_results/{trait_type}_flexrv_cct.mt'
    gene_mt = gene_mt.checkpoint(out_mt_path, overwrite=True)
    print(f"[flexrv_cct] Combined CCT result written -> {out_mt_path}")

    sig_ht = gene_mt.entries()
    sig_ht = sig_ht.filter(
        hl.any([
            (sig_ht.cct_p[sf] > 0) & (sig_ht.cct_p[sf] < sig_threshold)
            for sf in score_fields
        ])
    )
    sig_ht = sig_ht.select('phenocode', 'coding', 'trait_type', 'modifier', 'n_var', 'cct_p')

    out_tsv_path = f'{FLEXRV_BASE_PATH}{trait_type}_flexrv_cct.tsv'
    sig_ht.export(out_tsv_path)
    print(f"[flexrv_cct] Significant (p < {sig_threshold:.1e}) rows exported -> {out_tsv_path}")

    return gene_mt



# ── Core single weight burden strategy ──────────────────────────────────────────────

def run_all_models_batched_quant(mt, weight_fields, top_pcts=[0.05, 0.10, 0.15, 0.3, 0.5], weighted=True):
    """
    AC-weighted collapsing burden test for quantitative traits, swept across
    weight_fields x top_pcts.

    Convention: `top_pcts` values are the fraction of top-scoring variants to
    retain per weight field (e.g. 0.15 -> keep the top 15% of variants by
    score). Internally this is implemented as `threshold = 1.0 - top_pct`,
    i.e. a variant passes when `mt[w] >= threshold`. Matches
    run_all_models_batched_bin — the same top_pcts list means the same thing
    in both functions.

    weighted : bool, default True
        If True (original behavior), non-baseline scores are continuous in
        [0, 1] and are used directly as the AC-weighted burden weight w_i
        for any variant passing threshold. If False, the threshold is still
        used to select which variants are included, but every passing
        variant gets a binary weight of 1 instead of its score — i.e. this
        becomes an unweighted / count-based burden test. `genebass_baseline`
        is unaffected by this flag either way, since it already always uses
        a weight of 1. Matches the `weighted` flag in
        run_all_models_batched_bin.

    'genebass_baseline' (score identically 1.0, no thresholding) is handled
    once rather than swept across top_pcts, since sweeping it would just
    recompute the same "keep everything" aggregation N times.
    """
    # mt = mt.repartition(14000)

    thresholds = {p: 1.0 - p for p in top_pcts}
    non_baseline = [w for w in weight_fields if w != "genebass_baseline"]
    has_baseline = "genebass_baseline" in weight_fields

    mt = mt.filter_rows(
        (hl.is_defined(mt["genebass_baseline"]) if has_baseline else hl.bool(False)) |
        hl.any(*[
            hl.is_defined(mt[w]) & (mt[w] >= thresh)
            for w in non_baseline
            for thresh in thresholds.values()
        ])
    )

    # mt = mt.filter_entries(
    #     hl.is_defined(mt.BETA) & hl.is_defined(mt.SE) &
    #     (mt.SE > 0) & hl.is_defined(mt.AC) & (mt.AC > 0)
    # )

    agg_dict = {}

    # sigma^2_y (phenotype residual variance) is weight- and threshold-independent
    # (AC * SE^2 is ~constant across rare variants) — computed once and shared.
    agg_dict['_sigma2_y'] = hl.agg.mean(mt.AC * (mt.SE ** 2))

    if has_baseline:
        w_eff = hl.or_missing(hl.is_defined(mt['genebass_baseline']), mt['genebass_baseline'])
        agg_dict['_sn_genebass_baseline']   = hl.agg.sum(mt.AC * w_eff * mt.BETA)
        agg_dict['_sd_genebass_baseline']   = hl.agg.sum(mt.AC * (w_eff ** 2))
        agg_dict['n_var_genebass_baseline'] = hl.agg.count_where(hl.is_defined(w_eff))

    for w in non_baseline:
        for p, thresh in thresholds.items():
            tag = f'{w}__top{p}'
            passes = hl.is_defined(mt[w]) & (mt[w] >= thresh)
            if weighted:
                w_eff = hl.or_missing(passes, mt[w])
            else:
                w_eff = hl.or_missing(passes, 1)

            agg_dict[f'_sn_{tag}']   = hl.agg.sum(mt.AC * w_eff * mt.BETA)
            agg_dict[f'_sd_{tag}']   = hl.agg.sum(mt.AC * (w_eff ** 2))
            agg_dict[f'n_var_{tag}'] = hl.agg.count_where(hl.is_defined(w_eff))

    gene_mt = mt.group_rows_by(mt.gene).aggregate(**agg_dict)
    gene_mt = gene_mt.checkpoint(
        'gs://aou_amc/data/scallion/genebass/burden_results/allmodels_burden_qt_tmp.mt',
        overwrite=True,
    )

    s2y         = gene_mt['_sigma2_y']
    valid_sigma = hl.is_defined(s2y) & (s2y > 0)

    annot = {}

    def _add_annotations(tag):
        sn_f = gene_mt[f'_sn_{tag}']
        sd_f = gene_mt[f'_sd_{tag}']

        valid = valid_sigma & hl.is_defined(sd_f) & (sd_f > 0)

        annot[f'beta_{tag}'] = hl.if_else(valid, sn_f / sd_f, hl.missing(hl.tfloat64))
        annot[f'se_{tag}']   = hl.if_else(valid, hl.sqrt(s2y / sd_f), hl.missing(hl.tfloat64))
        annot[f'z_{tag}']    = hl.if_else(valid, sn_f / hl.sqrt(s2y * sd_f), hl.missing(hl.tfloat64))
        annot[f'p_{tag}']    = hl.if_else(
            valid,
            hl.pchisqtail((sn_f / hl.sqrt(s2y * sd_f)) ** 2, 1.0),
            hl.missing(hl.tfloat64),
        )

    if has_baseline:
        _add_annotations('genebass_baseline')
    for w in non_baseline:
        for p in top_pcts:
            _add_annotations(f'{w}__top{p}')

    gene_mt = gene_mt.annotate_entries(**annot)

    drop_fields = (
        ['_sigma2_y'] +
        (['_sn_genebass_baseline', '_sd_genebass_baseline'] if has_baseline else []) +
        [f'_sn_{w}__top{p}' for w in non_baseline for p in top_pcts] +
        [f'_sd_{w}__top{p}' for w in non_baseline for p in top_pcts]
    )

    return gene_mt.drop(*drop_fields)

def run_all_models_batched_bin(mt, weight_fields, top_pcts=[0.05, 0.10, 0.15, 0.3, 0.5], weighted=True):
    """
    IVW burden test for binary/categorical traits, swept across
    weight_fields x top_pcts.

    Convention: `top_pcts` values are the fraction of top-scoring variants to
    retain per weight field (e.g. 0.15 -> keep the top 15% of variants by
    score). Internally this is implemented as `threshold = 1.0 - top_pct`,
    i.e. a variant passes when `mt[w] >= threshold`. Matches
    run_all_models_batched_quant — the same top_pcts list means the same
    thing in both functions.

    weighted : bool, default True
        If True (original behavior), non-baseline scores are continuous in
        [0, 1] and are used directly as the IVW weight w_i for any variant
        passing threshold. If False, the threshold is still used to select
        which variants are included, but every passing variant gets a
        binary weight of 1 instead of its score — i.e. this becomes an
        unweighted / count-based burden test (equivalent to summing
        BETA_i / SE_i^2 over passing variants, with the denominator equal
        to a simple count of 1 / SE_i^2 rather than a score-weighted sum).
        `genebass_baseline` is unaffected by this flag either way, since it
        already always uses a weight of 1.

    For a variant passing threshold with weight w_i, this test combines
    per-variant score statistics U_i = BETA_i / SE_i^2 (with Var(U_i) =
    1/SE_i^2) as:
        U_burden   = sum(w_i * U_i)            = sum(w_i * BETA_i / SE_i^2)
        Var(U_burden) = sum(w_i^2 * Var(U_i))  = sum(w_i^2 / SE_i^2)
        Z = U_burden / sqrt(Var(U_burden))
    The denominator must be weighted by w_i**2 (not a bare pass/fail count)
    whenever weights are continuous rather than binary (i.e. whenever
    weighted=True).
    """

    # mt = mt.repartition(17000)

    thresholds = {p: 1.0 - p for p in top_pcts}
    non_baseline = [w for w in weight_fields if w != "genebass_baseline"]
    has_baseline = "genebass_baseline" in weight_fields

    # Keep a row if genebass_baseline is defined (always keep)
    # OR any non-baseline weight passes any threshold
    mt = mt.filter_rows(
        (hl.is_defined(mt["genebass_baseline"]) if has_baseline else hl.bool(False)) |
        hl.any(*[
            hl.is_defined(mt[w]) & (mt[w] >= thresh)
            for w in non_baseline
            for thresh in thresholds.values()
        ])
    )

    agg_dict = {}

    # genebass_baseline: one entry, weight identically 1.0, all defined variants included
    if has_baseline:
        # w_eff = hl.or_missing(hl.is_defined(mt["genebass_baseline"]), mt["genebass_baseline"])
        w_eff = hl.or_missing(hl.is_defined(mt["genebass_baseline"]), 1)
        agg_dict['sum_num_genebass_baseline']  = hl.agg.sum(w_eff * mt.BETA / (mt.SE ** 2))
        agg_dict['sum_info_genebass_baseline'] = hl.agg.sum((w_eff ** 2) / (mt.SE ** 2))
        agg_dict['n_var_genebass_baseline']    = hl.agg.count_where(hl.is_defined(w_eff))

    # Non-baseline weights: one entry per (weight, threshold) combination
    for w in non_baseline:
        for p, thresh in thresholds.items():
            tag = f'{w}__top{p}'
            passes = hl.is_defined(mt[w]) & (mt[w] >= thresh)
            if weighted:
                w_eff = hl.or_missing(passes, mt[w])
            else:
                w_eff = hl.or_missing(passes, 1)

            agg_dict[f'sum_num_{tag}']  = hl.agg.sum(w_eff * mt.BETA / (mt.SE ** 2))
            agg_dict[f'sum_info_{tag}'] = hl.agg.sum((w_eff ** 2) / (mt.SE ** 2))
            agg_dict[f'n_var_{tag}']    = hl.agg.count_where(hl.is_defined(w_eff))

    gene_mt = mt.group_rows_by(mt.gene).aggregate(**agg_dict)

    gene_mt = gene_mt.checkpoint(
        'gs://aou_amc/scallion/benchmark/data/genebass/ukb_weighted_burden/tmp/burden_bin_tmp.mt',
        overwrite=True,
    )

    annot = {}

    def _add_annotations(tag):
        si = gene_mt[f'sum_info_{tag}']
        sn = gene_mt[f'sum_num_{tag}']
        valid = hl.is_defined(si) & (si > 0)
        annot[f'beta_{tag}'] = hl.if_else(valid, sn / si,           hl.missing(hl.tfloat64))
        annot[f'se_{tag}']   = hl.if_else(valid, 1.0 / hl.sqrt(si), hl.missing(hl.tfloat64))
        annot[f'z_{tag}']    = hl.if_else(valid, sn / hl.sqrt(si),  hl.missing(hl.tfloat64))
        annot[f'p_{tag}']    = hl.if_else(
            valid,
            hl.pchisqtail((sn / hl.sqrt(si)) ** 2, 1.0),
            hl.missing(hl.tfloat64),
        )

    if has_baseline:
        _add_annotations('genebass_baseline')
    for w in non_baseline:
        for p in top_pcts:
            _add_annotations(f'{w}__top{p}')

    gene_mt = gene_mt.annotate_entries(**annot)

    drop_fields = (
        (['sum_num_genebass_baseline', 'sum_info_genebass_baseline'] if has_baseline else []) +
        [f'sum_num_{w}__top{p}'  for w in non_baseline for p in top_pcts] +
        [f'sum_info_{w}__top{p}' for w in non_baseline for p in top_pcts]
    )

    return gene_mt.drop(*drop_fields)



def run_temporary():
    var_path    = 'gs://ukbb-exome-public/500k/results/variant_results.mt'
    mt_genebass = hl.read_matrix_table(var_path)
    ht_scallion = hl.read_table('gs://aou_amc/data/scallion/genebass/predictions/preds_by_chrom/all_chr_pct_preds.ht')
    mt_genebass = filter_scallion_data(mt_genebass)
    mt_genebass = mt_genebass.filter_rows(hl.is_defined(ht_scallion[mt_genebass.locus, mt_genebass.alleles]))

    mt_genebass = mt_genebass.filter_cols(
        hl.literal(["icd10", "categorical"]).contains(mt_genebass.trait_type) & 
        (mt_genebass.modifier != "custom")
    )

    mt_genebass = mt_genebass.filter_entries(
        hl.is_defined(mt_genebass.BETA) &
        hl.is_defined(mt_genebass.SE) &
        hl.is_defined(mt_genebass.AC) &
        (mt_genebass.AC >= 1) & (mt_genebass.AC <= 100) &
        (mt_genebass.Pvalue < 2.5e-6)
    )

    mt_genebass = mt_genebass.filter_rows(
        hl.agg.any(hl.is_defined(mt_genebass.BETA) )
    )

    ht_sig_variants = mt_genebass.rows()

    ht_sig_variants = ht_sig_variants.checkpoint(
        'gs://aou_amc/data/scallion/genebass/genebass_wscallion_significant_vars.ht',
        overwrite = True
    )


# ── FlexRV burden strategy ──────────────────────────────────────────────
def run_flexrv_burden_bin(
    mt:           hl.MatrixTable,
    score_fields: List[str],
    maf_field:    str = 'AF',
) -> hl.MatrixTable:
    
    # mt = mt.filter_entries(
    #     hl.is_defined(mt.BETA) & hl.is_defined(mt.SE) & (mt.SE > 0) & 
    #     hl.is_defined(mt.AC) & (mt.AC >= 3) & (mt.AC <= 20)
    # )

    mt = mt.repartition(14000)

    n_samples = hl.int32(mt.n_cases) + hl.int32(mt.n_controls)
    inv_var   = 1.0 / (mt.SE ** 2)
    w_dict    = _entry_weight_dict(mt, score_fields, maf_field, n_samples)

    gene_mt = mt.group_rows_by(mt.gene).aggregate(
        _sum_num  = hl.struct(**{
            sf: hl.agg.array_sum(
                w_dict[sf].map(lambda w: w * mt.BETA * inv_var)
            )
            for sf in score_fields
        }),
        _sum_info = hl.struct(**{
            sf: hl.agg.array_sum(
                w_dict[sf].map(lambda w: (w ** 2) * inv_var)
            )
            for sf in score_fields
        }),
        n_var = hl.agg.count(),
    )
    gene_mt = gene_mt.checkpoint(
        'gs://aou_amc/data/scallion/genebass/burden_results/flexrv_burden_bin_tmp.mt',
        overwrite=True,
    )

    gene_mt = gene_mt.annotate_entries(
        z_arr = hl.struct(**{
            sf: hl.range(N_WEIGHTS).map(
                lambda i: hl.if_else(
                    hl.is_defined(gene_mt._sum_info[sf][i]) & (gene_mt._sum_info[sf][i] > 0),
                    gene_mt._sum_num[sf][i] / hl.sqrt(gene_mt._sum_info[sf][i]),
                    hl.missing(hl.tfloat64),
                )
            )
            for sf in score_fields
        }),
        p_arr = hl.struct(**{
            sf: hl.range(N_WEIGHTS).map(
                lambda i: hl.if_else(
                    hl.is_defined(gene_mt._sum_info[sf][i]) & (gene_mt._sum_info[sf][i] > 0),
                    hl.pchisqtail(
                        (gene_mt._sum_num[sf][i] / hl.sqrt(gene_mt._sum_info[sf][i])) ** 2,
                        1.0,
                    ),
                    hl.missing(hl.tfloat64),
                )
            )
            for sf in score_fields
        }),
    )
    return gene_mt.drop('_sum_num', '_sum_info')

def run_flexrv_burden_quant(
    mt:           hl.MatrixTable,
    score_fields: List[str],
    maf_field:    str = 'AF',
) -> hl.MatrixTable:
    """
    Quantitative burden — n_samples = n_cases (total N for quant traits in GeneBass).
    sigma²_y = mean(AC · SE²) is weight-independent, computed once as a scalar.
    Output entry fields: z_arr, p_arr
        Struct keyed by score field name, each value is an array of length
        N_WEIGHTS. e.g. z_arr.my_score[i], p_arr.my_score[i].
    Use get_flexrv_weight_keys(score_fields) to label each index within
    a score's array.
    """
    # mt = mt.filter_entries(
    #     hl.is_defined(mt.BETA) & hl.is_defined(mt.SE) &
    #     (mt.SE > 0) & hl.is_defined(mt.AC) & (mt.AC >= 3) & (mt.AC <= 20)
    # )

    n_samples = hl.int32(mt.n_cases) + hl.int32(mt.n_controls)
    ac        = hl.float64(mt.AC)
    w_dict    = _entry_weight_dict(mt, score_fields, maf_field, n_samples)

    gene_mt = mt.group_rows_by(mt.gene).aggregate(
        _sum_num   = hl.struct(**{
            sf: hl.agg.array_sum(
                w_dict[sf].map(lambda w: ac * w * mt.BETA)
            )
            for sf in score_fields
        }),
        _sum_denom = hl.struct(**{
            sf: hl.agg.array_sum(
                w_dict[sf].map(lambda w: ac * (w ** 2))
            )
            for sf in score_fields
        }),
        _sigma2_y  = hl.agg.mean(ac * (mt.SE ** 2)),  # scalar; weight- and threshold-independent, over all variants
        n_var      = hl.agg.count(),
    )
    gene_mt = gene_mt.checkpoint(
        'gs://aou_amc/data/scallion/genebass/burden_results/flexrv_burden_qt_tmp.mt',
        overwrite=True,
    )

    s2y         = gene_mt._sigma2_y
    valid_sigma = hl.is_defined(s2y) & (s2y > 0)

    gene_mt = gene_mt.annotate_entries(
        z_arr = hl.struct(**{
            sf: hl.range(N_WEIGHTS).map(
                lambda i: hl.if_else(
                    valid_sigma & hl.is_defined(gene_mt._sum_denom[sf][i]) & (gene_mt._sum_denom[sf][i] > 0),
                    gene_mt._sum_num[sf][i] / hl.sqrt(s2y * gene_mt._sum_denom[sf][i]),
                    hl.missing(hl.tfloat64),
                )
            )
            for sf in score_fields
        }),
        p_arr = hl.struct(**{
            sf: hl.range(N_WEIGHTS).map(
                lambda i: hl.if_else(
                    valid_sigma & hl.is_defined(gene_mt._sum_denom[sf][i]) & (gene_mt._sum_denom[sf][i] > 0),
                    hl.pchisqtail(
                        (gene_mt._sum_num[sf][i] / hl.sqrt(s2y * gene_mt._sum_denom[sf][i])) ** 2,
                        1.0,
                    ),
                    hl.missing(hl.tfloat64),
                )
            )
            for sf in score_fields
        }),
    )
    return gene_mt.drop('_sum_num', '_sum_denom', '_sigma2_y')


# ── FlexRV phenotype selection ────────────────────────────────────────────────
# Picks the phenocodes (bin: + coding) where pLoF burden testing found real
# signal that the current AM / scallion-baseline scores missed — the panel
# where a FlexRV rerun has headroom to show a power gain. See
# select_phenotypes.py for the original, more exploratory version of this.
UKB_BURDEN_BASE_PATH    = "gs://aou_amc/scallion/benchmark/results/genebass/ukb_weighted_burden"
FIGURES_BASE_PATH       = f"{UKB_BURDEN_BASE_PATH}/figures"
FLEXRV_BASE_PATH        = f"{UKB_BURDEN_BASE_PATH}/flexrv/"
# Phenotype lists are inputs to FlexRV, not results, so they live under the
# data/ tree rather than alongside FLEXRV_BASE_PATH's burden result outputs.
FLEXRV_PHENOS_BASE_PATH = "gs://aou_amc/scallion/benchmark/data/genebass/flexrv/"
FLEXRV_PHENOS_PATH      = {
    'bin': f'{FLEXRV_PHENOS_BASE_PATH}phenotypes_flexRV/binary_phenos.tsv',
    'qt':  f'{FLEXRV_PHENOS_BASE_PATH}phenotypes_flexRV/qt_phenos.tsv',
}
# Score fields + MAF field the FlexRV weight grid is built from (--run_burden
# --burden_mode flexrv) and later combined across (combine_flexrv_cct).
FLEXRV_PRIMARY_SCORE_FIELDS: List[str] = [
    'AM_pct',
    'pred_scallion_prob_mixture_clinvar_multi_drop_conflicting_pct',
]
FLEXRV_MAF_FIELD = 'AF.Cases'
FLEXRV_BATCH_OUT_PREFIX = {'bin': 'flexrv_burden_bin', 'qt': 'flexrv_burden_qt'}

PHENO_SELECT_SIG_THRESHOLD = 2.5e-6
PHENO_SELECT_N             = 100
PHENO_SELECT_MODEL_HINTS   = [
    "p_AM_pct__top0.85",
    "p_pred_scallion_prob_mixture_new_default_baseline_random_forest_regressor_pct__top0.85",
]
CODING_NA_SENTINEL = "__NA_CODING__"

FIG_WIDTH_IN      = 18
BAR_HEIGHT_IN     = 0.32   # vertical space per bar, so labels don't overlap regardless of bin size
MIN_ROW_HEIGHT_IN = 2.5
N_TOP_LABELED     = 3      # how many top-overlap scatter points get their method name written next to them


def _short_label(col, suffix):
    col = col.replace(f"__{suffix}", "")
    col = col.removeprefix("p_pred_scallion_").removeprefix("p_")
    col = col.removesuffix("_prob_pct").removesuffix("_pred_pct").removesuffix("_pct")
    col = col.replace("mixture_", "").replace("new_default_", "")
    return col.replace("_", " ").strip()


def _bar_color(col):
    # Both scallion families are regressors (e.g. XGBoost), not classifiers —
    # "p_pred_scallion_llr_..." (log-likelihood-ratio) vs
    # "p_pred_scallion_prob_..." (probability-mixture) just names which
    # score the regressor was trained to output, not binary vs. continuous.
    if "genebass_baseline" in col: return "#E24B4A"
    if "scallion" in col and "prob" in col: return "#378ADD"
    if "scallion" in col and "llr"  in col: return "#1D9E75"
    return "#888780"


def _load_missense_and_lof_reference(tsv_path, exclude_custom=True):
    """
    Shared prep for phenotype selection and the overlap plot: load a
    summarize-step TSV, split off the pLoF-significant reference set, and
    return the score-based (sc_ht-derived) rows deduped by key. The join
    that produces this TSV duplicates each sc_ht row across every GeneBass
    annotation (pLoF/missense|LC/synonymous) independently significant for
    the same (gene, phenocode[, coding]) — filtering on `annotation` would
    silently drop rows whose only GeneBass match is pLoF, so we dedupe on
    the key instead. Auto-detects whether 'coding' is part of the (gene,
    phenocode[, coding]) matching key: qt files leave 'coding' always NA
    (not part of the key); bin files populate it to distinguish case
    definitions within a phenocode (part of the key).

    exclude_custom=False keeps custom phenocodes/modifiers in — used for the
    "_with_customphenos" overlap figure, which reports on all phenotypes.

    Returns (df, triplet_cols, lof_triplet_df, sig_mask).
    """
    df = pd.read_csv(tsv_path, sep="\t")
    if exclude_custom:
        df = df[df["modifier"] != "custom"]
        df = df[~df["phenocode"].str.endswith("custom")]

    use_coding = ("coding" in df.columns) and df["coding"].notna().any()
    if "coding" in df.columns:
        df["coding"] = df["coding"].fillna(CODING_NA_SENTINEL)
    triplet_cols = ["gene_symbol", "phenocode", "coding"] if use_coding else ["gene_symbol", "phenocode"]

    def sig_mask(frame, col):
        return (frame[col] > 0) & (frame[col] < PHENO_SELECT_SIG_THRESHOLD)

    plof_df        = df[df["annotation"] == "pLoF"]
    lof_triplet_df = plof_df.loc[sig_mask(plof_df, "Pvalue_Burden"), triplet_cols].drop_duplicates()

    # Restrict to the two annotations we care about (or no GeneBass match at
    # all, i.e. a pure sc_ht-only hit) *before* deduping — otherwise, for a
    # key duplicated across an unwanted annotation (synonymous, the combined
    # pLoF|missense|LC) and a wanted one, drop_duplicates could arbitrarily
    # keep the unwanted row depending on TSV row order.
    df = df[df["annotation"].isin(["pLoF", "missense|LC"]) | df["annotation"].isna()]
    p_cols = [c for c in df.columns if c.startswith("p_")]
    df = df.dropna(subset=p_cols, how="all").drop_duplicates(subset=triplet_cols).copy()
    return df, triplet_cols, lof_triplet_df, sig_mask


def select_phenotypes_for_flexrv(trait_type, burden_mode="multithreshold_weighted"):
    """
    Reads the 'summarize' step's TSV for `trait_type`/`burden_mode`, selects
    up to PHENO_SELECT_N phenocodes with pLoF signal missed by the current
    baseline scores, and writes:
      - the (phenocode, coding) pairs, straight to the path run_flexrv_burden_*
        already reads phenotypes from (FLEXRV_PHENOS_PATH[trait_type])
      - a per-phenocode recovery-stats CSV, for reference
    """
    tsv_path = f"{UKB_BURDEN_BASE_PATH}/{trait_type}_{burden_mode}.tsv"
    print(f"[select_phenos_flexrv] Loading '{trait_type}' results from {tsv_path}...")
    df, triplet_cols, lof_triplet_df, sig_mask = _load_missense_and_lof_reference(tsv_path)
    use_coding     = "coding" in triplet_cols
    match_key_cols = [c for c in triplet_cols if c != "phenocode"]
    print(f"[select_phenos_flexrv] triplet_cols: {triplet_cols}  |  "
          f"pLoF significant tuples: {len(lof_triplet_df)}")

    def resolve_column(hint, columns):
        if hint in columns:
            return hint
        pat = re.compile(rf'(^|_){re.escape(hint)}(_|$)')
        matches = [c for c in columns if pat.search(c)]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"Model hint {hint!r} did not resolve to exactly one column (matches={matches}).")

    model_cols = {hint: resolve_column(hint, df.columns) for hint in PHENO_SELECT_MODEL_HINTS}

    def sig_match_keys_by_phenocode(frame, col):
        sub = frame.loc[sig_mask(frame, col), ["phenocode"] + match_key_cols].drop_duplicates()
        if sub.empty:
            return {}
        return (
            sub.groupby("phenocode")[match_key_cols]
            .apply(lambda g: set(g.itertuples(index=False, name=None)))
            .to_dict()
        )

    model_sig_by_pheno = {hint: sig_match_keys_by_phenocode(df, col) for hint, col in model_cols.items()}
    lof_by_pheno = (
        lof_triplet_df.groupby("phenocode")[match_key_cols]
        .apply(lambda g: set(g.itertuples(index=False, name=None)))
        .to_dict()
        if not lof_triplet_df.empty else {}
    )

    records = []
    for pheno, lof_keys in lof_by_pheno.items():
        row = {"phenocode": pheno, "n_lof_sig_genes": len(lof_keys)}
        recovered_union = set()
        for hint in PHENO_SELECT_MODEL_HINTS:
            recovered = lof_keys & model_sig_by_pheno[hint].get(pheno, set())
            row[f"n_recovered__{hint}"] = len(recovered)
            recovered_union |= recovered
        row["n_missed"]    = len(lof_keys) - len(recovered_union)
        row["frac_missed"] = row["n_missed"] / len(lof_keys)
        records.append(row)

    candidates = pd.DataFrame(records)
    candidates = candidates[candidates["n_missed"] > 0].sort_values(
        ["n_missed", "frac_missed"], ascending=[False, False]
    ).reset_index(drop=True)

    selected_df = candidates.head(PHENO_SELECT_N).copy()
    selected_phenocodes = selected_df["phenocode"].tolist()
    print(f"[select_phenos_flexrv] Selected {len(selected_df)}/{PHENO_SELECT_N} phenotypes "
          f"(of {len(candidates)} candidates with pLoF signal + >=1 missed gene).")

    stats_path = f"{UKB_BURDEN_BASE_PATH}/selected_phenotypes_{trait_type}.csv"
    selected_df.to_csv(stats_path, index=False)
    print(f"[select_phenos_flexrv] Saved selection stats -> {stats_path}")

    # ── Hand the selected (phenocode[, coding]) pairs straight to FlexRV ──────
    pairs = lof_triplet_df.loc[
        lof_triplet_df["phenocode"].isin(selected_phenocodes),
        ["phenocode", "coding"] if use_coding else ["phenocode"],
    ].drop_duplicates()
    if use_coding:
        pairs["coding"] = pairs["coding"].replace(CODING_NA_SENTINEL, "NA")
    else:
        pairs["coding"] = "NA"
    flexrv_path = FLEXRV_PHENOS_PATH[trait_type]
    pairs.to_csv(flexrv_path, sep="\t", index=False)
    print(f"[select_phenos_flexrv] Saved {len(pairs)} (phenocode, coding) pairs for FlexRV -> {flexrv_path}")

    return selected_df, pairs


def plot_missense_lof_overlap(trait_type, tsv_path):
    """
    Aggregated missense/pLoF overlap figures across all top-pct bins for
    `trait_type` (a row per bin: bar chart + discovery-size scatter), saved
    under FIGURES_BASE_PATH:
      - triplet-level: (gene_symbol, phenocode[, coding]) hits
      - gene-level:    gene_symbol hits, collapsed across phenotypes
      - triplet-level over ALL phenotypes, including the custom ones the
        two figures above exclude (missense_lof_overlap_triplet_{trait_type}_with_customphenos.png)

    Each row's height scales with its actual bar count (BAR_HEIGHT_IN per
    bar), so labels stay legible regardless of how many score columns a bin
    has. The top N_TOP_LABELED points by overlap are labeled directly on
    each scatter.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import fsspec

    genebass_col = "p_genebass_baseline"
    legend_patches = [
        mpatches.Patch(color="#E24B4A", label="Genebass baseline"),
        mpatches.Patch(color="#378ADD", label="Scallion (prob mixture)"),
        mpatches.Patch(color="#1D9E75", label="Scallion (LLR)"),
        mpatches.Patch(color="#888780", label="Pathogenicity scores"),
    ]

    def render(level, df, triplet_cols, lof_triplet_df, sig_mask, filename_suffix=""):
        all_p_cols = [c for c in df.columns if c.startswith("p_")]
        bin_suffixes = sorted(
            {m.group(1) for c in all_p_cols if (m := re.search(r'__(top[\d.]+)$', c))},
            key=lambda x: float(x.replace("top", "")),
        )

        if level == "triplet":
            ref_set, key_cols, overlap_label = set(lof_triplet_df.itertuples(index=False, name=None)), triplet_cols, "triplets"
        elif level == "gene":
            ref_set, key_cols, overlap_label = set(lof_triplet_df["gene_symbol"]), "gene_symbol", "genes"
        else:
            raise ValueError(f"Unknown level: {level}")

        def sig_items(frame, col):
            mask = sig_mask(frame, col)
            if isinstance(key_cols, str):
                return set(frame.loc[mask, key_cols])
            return set(frame.loc[mask, key_cols].drop_duplicates().itertuples(index=False, name=None))

        print(f"[plot_missense_lof_overlap] [{level}{filename_suffix}] pLoF significant reference {overlap_label}: {len(ref_set)}")

        per_suffix_results = {}
        for suffix in bin_suffixes:
            p_cols = [c for c in all_p_cols if c.endswith(f"__{suffix}")]
            if genebass_col in df.columns and genebass_col not in p_cols:
                p_cols = [genebass_col] + p_cols

            records = []
            for col in p_cols:
                items = sig_items(df, col)
                records.append({
                    "label":      _short_label(col, suffix),
                    "n_missense": len(items),
                    "n_overlap":  len(items & ref_set),
                    "color":      _bar_color(col),
                })
            per_suffix_results[suffix] = pd.DataFrame(records).sort_values(
                "n_overlap", ascending=False
            ).reset_index(drop=True)

        # Reserve a fixed number of INCHES (not a fraction) for the suptitle
        # + legend strip at the top and a small margin at the bottom, so
        # that margin doesn't balloon into a huge blank gap on a tall,
        # many-row figure (matplotlib's default top/bottom margins are
        # fractional, which looks fine on a normal figure but leaves a
        # multi-inch blank band once the figure is 40+ inches tall).
        top_margin_in, bottom_margin_in = 1.6, 0.3
        row_heights = [max(MIN_ROW_HEIGHT_IN, len(per_suffix_results[s]) * BAR_HEIGHT_IN) for s in bin_suffixes]
        fig_height = sum(row_heights) + top_margin_in + bottom_margin_in
        fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height))
        gs  = GridSpec(len(bin_suffixes), 2, figure=fig, height_ratios=row_heights,
                        top=1 - top_margin_in / fig_height, bottom=bottom_margin_in / fig_height,
                        hspace=0.35, wspace=0.25, width_ratios=[2, 1])

        for i, suffix in enumerate(bin_suffixes):
            results = per_suffix_results[suffix]

            ax_bar = fig.add_subplot(gs[i, 0])
            ax_bar.barh(range(len(results)), results["n_overlap"], color=results["color"],
                        height=0.7, edgecolor="white", linewidth=0.6)
            ax_bar.set_yticks(range(len(results)))
            ax_bar.set_yticklabels(results["label"], fontsize=10)
            ax_bar.invert_yaxis()
            ax_bar.set_title(f"bin: {suffix}  |  LoF reference {overlap_label}: {len(ref_set)}",
                              fontsize=12, fontweight="bold")
            ax_bar.spines[["top", "right"]].set_visible(False)

            ax_scatter = fig.add_subplot(gs[i, 1])
            ax_scatter.scatter(results["n_missense"], results["n_overlap"], c=results["color"],
                                s=60, edgecolors="white", linewidths=0.6)
            for _, row in results.nlargest(N_TOP_LABELED, "n_overlap").iterrows():
                ax_scatter.annotate(
                    row["label"], xy=(row["n_missense"], row["n_overlap"]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color="#444441", va="center",
                )
            ax_scatter.set_xlabel(f"n significant missense {overlap_label}", fontsize=9)
            ax_scatter.set_ylabel("n overlapping pLoF", fontsize=9)
            ax_scatter.spines[["top", "right"]].set_visible(False)
            ax_scatter.grid(linestyle="--", linewidth=0.4, alpha=0.5)

        fig.legend(handles=legend_patches, loc="upper center", ncol=4, fontsize=11,
                   bbox_to_anchor=(0.5, 1 - 0.65 / fig_height))
        fig.suptitle(f"Missense / pLoF {level} overlap — {trait_type}{filename_suffix.replace('_', ' ')} — p < {PHENO_SELECT_SIG_THRESHOLD:.1e}",
                     fontsize=14, fontweight="bold", y=1 - 0.2 / fig_height)

        fig_path = f"{FIGURES_BASE_PATH}/missense_lof_overlap_{level}_{trait_type}{filename_suffix}.png"
        with fsspec.open(fig_path, "wb") as f:
            fig.savefig(f, dpi=180, bbox_inches="tight", format="png")
        plt.close(fig)
        print(f"[plot_missense_lof_overlap] Saved {level}{filename_suffix} overlap figure -> {fig_path}")

    print(f"[plot_missense_lof_overlap] Loading '{trait_type}' results from {tsv_path}...")
    df, triplet_cols, lof_triplet_df, sig_mask = _load_missense_and_lof_reference(tsv_path)
    render("triplet", df, triplet_cols, lof_triplet_df, sig_mask)
    render("gene", df, triplet_cols, lof_triplet_df, sig_mask)

    print(f"[plot_missense_lof_overlap] Loading '{trait_type}' results (all phenotypes) from {tsv_path}...")
    df_all, triplet_cols_all, lof_triplet_df_all, sig_mask_all = _load_missense_and_lof_reference(
        tsv_path, exclude_custom=False
    )
    render("triplet", df_all, triplet_cols_all, lof_triplet_df_all, sig_mask_all, filename_suffix="_with_customphenos")


# ── Utils minor processing ──────────────────────────────────────────────
# Util for some filtering
def filter_scallion_data(mt):
    '''Exclude data used to train scallion'''
    save_out_ht      = 'gs://aou_amc/scallion/data/pLoF_genebass_significant_nosparse.ht'
    scallion_training = hl.read_table(save_out_ht)

    exclude_phenos = scallion_training.key_by('phenocode').select()
    exclude_genes  = scallion_training.key_by('gene_symbol').select()

    mt = mt.filter_cols(hl.is_missing(exclude_phenos[mt.phenocode]))
    mt = mt.filter_rows(hl.is_missing(exclude_genes[mt.gene]))

    return mt


# ── Args ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run burden analysis and/or summarization for Genebass + Scallion"
    )
    parser.add_argument('--run_burden', action='store_true',
                        help='Run burden models')
    parser.add_argument('--summarize',  action='store_true',
                        help='Summarize and export significant results')
    parser.add_argument('--burden_mode',
                        choices=['multithreshold_weighted', 'multithreshold_unweighted', 'flexrv'],
                        default='multithreshold_weighted',
                        help=(
                            'Which burden pipeline to run (--run_burden) or summarize '
                            '(--summarize). multithreshold_weighted/unweighted use the '
                            'standard per-weight-field sweep (with or without continuous '
                            'score-based weights); flexrv uses the FlexRV 192-weight-grid '
                            'approach. Also sets the output/input path suffix under '
                            'UKB_BURDEN_BASE_PATH, so --run_burden and --summarize stay '
                            'pointed at the same data.'
                        ))
    parser.add_argument('--trait_type', choices=['qt', 'bin'], required=True,
                        help='Trait type to run: qt (quantitative/continuous) or bin (binary/categorical)')
    parser.add_argument('--test', action='store_true',
                        help='Run on test phenotype C50 only')
    parser.add_argument('--run_tmp', action='store_true',
                        help='Run temp code filter significant')
    parser.add_argument('--overwrite_pct_ht', action='store_true',
                        help=(
                            'Force reimport of the Scallion pct TSV and overwrite the '
                            'checkpointed HT, even if it already exists.'
                        ))
    parser.add_argument('--select_phenos_flexrv', action='store_true',
                        help=(
                            'Select phenotypes for the FlexRV rerun from the --trait_type '
                            'summarize TSV: phenocodes with pLoF burden signal missed by '
                            'the current baseline scores. Writes the (phenocode, coding) '
                            'pairs straight to the FlexRV phenotype list path, plus a '
                            'stats CSV.'
                        ))
    parser.add_argument('--overwrite_summary', action='store_true',
                        help=(
                            'Force --summarize to rebuild and re-export the joined significant-'
                            'results TSV even if it already exists. If not set and the TSV '
                            'already exists, the join/export step is skipped (the overlap '
                            'figure is still (re)generated from the existing TSV).'
                        ))
    parser.add_argument('--combine_flexrv_cct', action='store_true',
                        help=(
                            'Merge the batched --run_burden --burden_mode flexrv checkpoints '
                            'for --trait_type and collapse the 192-weight grid into one CCT '
                            'p-value per gene x phenotype x score field (see add_cct_p_entry '
                            '/ combine_flexrv_cct).'
                        ))
    return parser.parse_args()



def main(args):
    if args.run_tmp or args.run_burden or args.summarize or args.combine_flexrv_cct:
        hl.init_batch(
            billing_project="all-by-aou",
            remote_tmpdir='gs://aou_tmp/v8',   # note: remote_tmpdir, not tmp_dir, for the batch backend
            # worker_memory='10Gi',
            worker_memory='highmem',
            driver_memory='highmem',
            gcs_requester_pays_configuration="aou-neale-gwas",
        )

    if args.run_tmp:
        run_temporary()

    if args.run_burden:
        var_path    = 'gs://ukbb-exome-public/500k/results/variant_results.mt'
        mt_genebass = hl.read_matrix_table(var_path)
        
        ht_scallion_path = 'gs://aou_amc/scallion/benchmark/data/genebass_w_vsm_w_predictions_w_pct.ht'
        if hl.hadoop_exists(ht_scallion_path) and not args.overwrite_pct_ht:
            ht_scallion = hl.read_table(ht_scallion_path)
        else:
            ht_scallion = hl.import_table(
                'gs://aou_amc/scallion/benchmark/data/genebass_w_vsm_w_predictions_w_pct.tsv',
                impute=True,
            )
            ht_scallion = ht_scallion.annotate(
                locus=hl.parse_locus(ht_scallion.locus, reference_genome='GRCh38'),
                alleles=ht_scallion.alleles.replace(r'[\[\]"]', '').split(','),
            )
            ht_scallion = ht_scallion.key_by('locus', 'alleles')
            ht_scallion = ht_scallion.checkpoint(ht_scallion_path, overwrite=True)


        if args.burden_mode == 'flexrv':
            PRIMARY_SCORE_FIELDS = FLEXRV_PRIMARY_SCORE_FIELDS
            MAF_FIELD = FLEXRV_MAF_FIELD
            BATCH_SIZE = 80
            BASE_PATH = FLEXRV_BASE_PATH

            print("[flexRV] Starting flexRV burden pipeline...")
            ht_scallion = ht_scallion.select(*PRIMARY_SCORE_FIELDS)

            # --- Configure trait type ---
            trait_config = {
                'bin': (
                    hl.literal(['icd10', 'categorical']).contains(mt_genebass.trait_type),
                    FLEXRV_PHENOS_PATH['bin'],
                    run_flexrv_burden_bin,
                    FLEXRV_BATCH_OUT_PREFIX['bin'],
                ),
                'qt': (
                    mt_genebass.trait_type == 'continuous',
                    FLEXRV_PHENOS_PATH['qt'],
                    run_flexrv_burden_quant,
                    FLEXRV_BATCH_OUT_PREFIX['qt'],
                ),
            }

            # --- Filter columns to relevant phenotypes and trait type ---
            if args.test:
                print(f"[flexRV] Test mode: filtering to phenotype {test_pheno}...")
                test_pheno = 'C43'
                mt_genebass = mt_genebass.filter_cols(mt_genebass.phenocode == test_pheno)
                if mt_genebass.count_cols() == 0:
                    raise ValueError(
                        f"Test phenotype {test_pheno} not found after filtering — check trait_type or phenos file."
                    )
                mt_genebass = mt_genebass.repartition(100).checkpoint(
                    f'{BASE_PATH}/burden_results/flexrv_{test_pheno}_test_v2.mt', overwrite=True
                )
                print(f"[flexRV] Test checkpoint written for {test_pheno}.")
            else:
                if args.trait_type not in trait_config:
                    raise ValueError(f"Invalid trait type: {args.trait_type}")
                trait_type_filter, phenos_path, run_fn, out_prefix = trait_config[args.trait_type]
                print(f"[flexRV] Trait type: '{args.trait_type}' — loading phenotypes from {phenos_path}...")
                flexrv_phenos = hl.import_table(phenos_path).key_by('phenocode', 'coding')
                mt_genebass = mt_genebass.filter_cols(
                    hl.is_defined(flexrv_phenos[mt_genebass.phenocode, mt_genebass.coding]) &
                    trait_type_filter
                )
                print(f"[flexRV] Column filter applied (phenotype list + trait type).")

            # --- Filter and annotate rows with scallion scores ---
            print("[flexRV] Filtering and annotating rows with scallion scores...")
            scallion = ht_scallion[mt_genebass.row_key]
            mt_genebass = mt_genebass.filter_rows(
                hl.is_defined(scallion) | hl.or_else(mt_genebass.annotation == 'pLoF', False)
            )
            scallion = ht_scallion[mt_genebass.row_key]
            mt_genebass = mt_genebass.annotate_rows(
                **{f: scallion[f] for f in PRIMARY_SCORE_FIELDS}
            )

            # --- Filter entries to valid, AC-passing associations ---
            print("[flexRV] Filtering entries (BETA/SE/AC validity)...")
            mt_genebass = mt_genebass.filter_entries(
                hl.is_defined(mt_genebass.BETA) &
                hl.is_defined(mt_genebass.SE) & (mt_genebass.SE > 0) &
                hl.is_defined(mt_genebass.AC) &
                (mt_genebass.AC >= 1) & (mt_genebass.AC <= 100)
            )
            mt_genebass = mt_genebass.filter_rows(hl.agg.any(hl.is_defined(mt_genebass.BETA)))

            # --- Batch and run ---
            mt_genebass = mt_genebass.add_col_index('col_idx')
            n_cols = mt_genebass.count_cols()
            n_batches = math.ceil(n_cols / BATCH_SIZE)
            print(f"[flexRV] Total columns: {n_cols} — running {n_batches} batch(es) of up to {BATCH_SIZE}...")

            for i in range(n_batches):
                start_idx = i * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, n_cols)
                print(f"[flexRV] Batch {i + 1}/{n_batches} (cols {start_idx}–{end_idx - 1})...")

                mt_batch = mt_genebass.filter_cols(
                    (mt_genebass.col_idx >= start_idx) & (mt_genebass.col_idx < end_idx)
                )
                gene_mt_batch = run_fn(mt_batch, PRIMARY_SCORE_FIELDS, MAF_FIELD)
                gene_mt_batch.checkpoint(
                    f'{BASE_PATH}/burden_results/{out_prefix}_batch{i + 1}.mt', overwrite=True
                )
                print(f"[flexRV] Batch {i + 1}/{n_batches} complete — checkpoint written.")

            print(f"[flexRV] All {n_batches} batch(es) complete.")
        
        else:
            print(f"[standard] Starting standard burden pipeline ({args.burden_mode})...")
            BASE_PATH = UKB_BURDEN_BASE_PATH
            weighted  = args.burden_mode == 'multithreshold_weighted'

            # mt_genebass = filter_scallion_data(mt_genebass)
            mt_genebass = mt_genebass.filter_rows(hl.is_defined(ht_scallion[mt_genebass.locus, mt_genebass.alleles]))

            calibrated_scores = [f for f in ht_scallion.row if f.endswith('_pct')]
            ht_scallion = ht_scallion.select(*calibrated_scores)
            mt_genebass = mt_genebass.annotate_rows(**ht_scallion[mt_genebass.row_key])
            mt_genebass = mt_genebass.annotate_rows(genebass_baseline=hl.float64(1.0))
            calibrated_scores = calibrated_scores + ['genebass_baseline']

            print(calibrated_scores)
            print(f"[standard] Scallion data filtered and annotated ({len(calibrated_scores)} score fields).")

            if args.test:
                test_pheno = 'C43'
                print(f"[standard] Test mode: filtering to phenotype {test_pheno}...")
                mt_genebass = mt_genebass.filter_cols(mt_genebass.phenocode == test_pheno)
                if mt_genebass.count_cols() == 0:
                    raise ValueError(f"Test phenotype {test_pheno} not found after filtering — check filter_scallion_data.")
                mt_genebass = mt_genebass.repartition(100).checkpoint(
                    f'{BASE_PATH}/tmp/standard_{test_pheno.lower()}_test.mt', overwrite=True
                )
                print(f"[standard] Test checkpoint written for {test_pheno}.")
            
            
            # --- Select run function and output path by trait type ---
            trait_config = {
                'bin': (
                    mt_genebass.filter_cols(hl.literal(["icd10", "categorical"]).contains(mt_genebass.trait_type) & (mt_genebass.modifier != "custom")),
                    run_all_models_batched_bin,
                    f'{BASE_PATH}/bin_{args.burden_mode}.mt',
                ),
                'qt': (
                    mt_genebass.filter_cols(mt_genebass.trait_type == 'continuous'),
                    run_all_models_batched_quant,
                    f'{BASE_PATH}/qt_{args.burden_mode}.mt',
                ),
            }

            if args.trait_type not in trait_config:
                raise ValueError(f"Invalid trait type: {args.trait_type}")

            mt_filtered, run_fn, out_path = trait_config[args.trait_type]
            print(f"[standard] Running '{args.trait_type}' burden model...")
            top_pcts = [0.10, 0.25, 0.50, 0.75, 0.85]
            gene_mt = run_fn(mt_filtered, calibrated_scores, top_pcts=top_pcts, weighted=weighted)
            gene_mt = gene_mt.checkpoint(out_path, overwrite=True)
            print(f"[standard] Done — checkpoint written to {out_path}.")

    if args.summarize:
        # ── CONFIG ───────────────────────────────────────────────────────────────
        GENEBASS_PATH    = "gs://ukbb-exome-public/500k/results/results.mt"
        BASE_PATH = f"{UKB_BURDEN_BASE_PATH}/{args.trait_type}_{args.burden_mode}"
        WEIGHTED_RESULTS_PATH = f"{BASE_PATH}.mt"
        OUT_TSV          = f"{BASE_PATH}.tsv"
        OUT_HT           = f"{BASE_PATH}.ht"
        REKEY_GB_HT      = "gs://aou_amc/data/scallion/genebass/burden_results_v2/_tmp_gb_rekeyed.ht"
        REKEY_SC_HT      = "gs://aou_amc/data/scallion/genebass/burden_results_v2/_tmp_sc_rekeyed.ht"
        SIG_THRESHOLD    = PHENO_SELECT_SIG_THRESHOLD
        DROP_COLS = ['interval', 'markerIDs', 'description',
                     'description_more', 'coding_description', 'category']

        if args.overwrite_summary or not hl.hadoop_exists(OUT_TSV):
            # ── 1. BUILD & CHECKPOINT gb_ht ──────────────────────────────────────
            # Filter INSIDE the MT (before entries()) so only significant rows/entries
            # are flattened — avoids exploding the full cross-product.
            sc_results = hl.read_matrix_table(WEIGHTED_RESULTS_PATH)
            sc_phenocodes = sc_results.cols().key_by().select('phenocode', 'coding')
            sc_phenocode_set = hl.literal(
                sc_phenocodes.aggregate(hl.agg.collect_as_set(sc_phenocodes.phenocode))
            )

            gene_mt = hl.read_matrix_table(GENEBASS_PATH)
            gene_mt = gene_mt.filter_cols(
                sc_phenocode_set.contains(gene_mt.phenocode)
            )
            gene_mt = gene_mt.filter_rows(
                hl.agg.any(
                    (gene_mt.Pvalue_Burden > 0) & (gene_mt.Pvalue_Burden < SIG_THRESHOLD)
                )
            )
            gene_mt = gene_mt.filter_entries(
                (gene_mt.Pvalue_Burden > 0) & (gene_mt.Pvalue_Burden < SIG_THRESHOLD)
            )

            gb_ht = (
                gene_mt.entries()
                .drop(*DROP_COLS)
                .key_by('gene_symbol', 'phenocode', 'coding')
                .select('annotation', 'trait_type', 'pheno_sex', 'modifier', 'Pvalue_Burden')
                .checkpoint(REKEY_GB_HT, overwrite=True)
            )

            # ── 2. BUILD & CHECKPOINT sc_ht ──────────────────────────────────────
            def make_entry_sig(mt):
                return hl.any([
                    (mt.entry[f] > 0) & (mt.entry[f] < SIG_THRESHOLD)
                    for f in p_fields
                ])

            p_fields = [f for f in sc_results.entry if f.startswith('p_')]

            # Build a significance flag across all p_scallion_* fields at entry level
            sc_results = sc_results.filter_rows(hl.agg.any(make_entry_sig(sc_results)))
            sc_results = sc_results.filter_entries(make_entry_sig(sc_results))

            sc_ht = sc_results.entries().rename({'gene': 'gene_symbol'})
            sc_ht = (
                sc_ht
                .key_by('gene_symbol', 'phenocode', 'coding')
                .select(*p_fields)
                .checkpoint(REKEY_SC_HT, overwrite=True)
            )

            # ── 3. JOIN ───────────────────────────────────────────────────────────
            # Both sides are now tiny — the join and shuffle are cheap
            joined_ht = gb_ht.join(sc_ht, how='outer')

            # ── 4. WRITE OUTPUT ───────────────────────────────────────────────────
            joined_ht = joined_ht.naive_coalesce(200).checkpoint(OUT_HT, overwrite=True)
            joined_ht.export(OUT_TSV)
            print(f"Significant rows written → {OUT_HT}")
            print(f"TSV exported             → {OUT_TSV}")
        else:
            print(f"[summarize] {OUT_TSV} already exists — skipping join/export "
                  f"(use --overwrite_summary to force).")

        # ── 5. VISUALIZE ─────────────────────────────────────────────────────────
        plot_missense_lof_overlap(args.trait_type, OUT_TSV)

    if args.select_phenos_flexrv:
        select_phenotypes_for_flexrv(args.trait_type, args.burden_mode)

    if args.combine_flexrv_cct:
        combine_flexrv_cct(args.trait_type)


if __name__ == '__main__':
    main(parse_args())


