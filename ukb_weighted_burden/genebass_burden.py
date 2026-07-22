import hail as hl
import argparse
import numpy as np
import pandas as pd
import math
from typing import List, Optional


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
    field1, field2 = maf_field.split('.')
    maf_expr = hl.coalesce(hl.float64(mt[field1][field2]), 0.0)

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
def add_cct_p_entry(
    gene_mt:      hl.MatrixTable,
    score_fields: List[str],
) -> hl.MatrixTable:
    """
    CCT across the 192-weight axis for each (gene, phenotype) entry.

    Expects entry field:
        p_arr: struct{ score_field: array<float64>[N_WEIGHTS] }

    Adds entry field:
        cct_p: struct{ score_field: float64 }
            One CCT p-value per gene × phenotype × score field.
    """
    pi = hl.float64(math.pi)

    final = {}
    for sf in score_fields:
        p_arr = gene_mt.p_arr[sf]   # array<float64>[N_WEIGHTS]

        cauchy_terms = p_arr.map(
            lambda p: hl.if_else(
                hl.is_defined(p),
                hl.tan(
                    (hl.float64(0.5) - hl.max(hl.float64(1e-15), hl.min(hl.float64(1.0), p)))
                    * pi
                ),
                hl.missing(hl.tfloat64),
            )
        )

        valid_terms = cauchy_terms.filter(hl.is_defined)
        n_valid     = hl.len(valid_terms)

        final[sf] = hl.if_else(
            n_valid > 0,
            hl.float64(0.5) - hl.atan(hl.sum(valid_terms) / hl.float64(n_valid)) / pi,
            hl.missing(hl.tfloat64),
        )

    return gene_mt.annotate_entries(
        cct_p=hl.struct(**final)
    )


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


# ── Core single weight burden strategy ──────────────────────────────────────────────

def run_all_models_batched_quant(mt, weight_fields, top_pcts=[0.05, 0.10, 0.15, 0.3, 0.5]):
    """
    AC-weighted collapsing burden test for quantitative traits, swept across
    weight_fields x top_pcts.

    Convention: `top_pcts` values are the fraction of top-scoring variants to
    retain per weight field (e.g. 0.15 -> keep the top 15% of variants by
    score). Internally this is implemented as `threshold = 1.0 - top_pct`,
    i.e. a variant passes when `mt[w] >= threshold`. Matches
    run_all_models_batched_bin — the same top_pcts list means the same thing
    in both functions.

    'genebass_baseline' (score identically 1.0, no thresholding) is handled
    once rather than swept across top_pcts, since sweeping it would just
    recompute the same "keep everything" aggregation N times.
    """
    mt = mt.repartition(14000)

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
            w_eff = hl.or_missing(hl.is_defined(mt[w]) & (mt[w] >= thresh), mt[w])
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

def run_all_models_batched_bin(mt, weight_fields, top_pcts=[0.05, 0.10, 0.15, 0.3, 0.5]):
    """
    IVW burden test for binary/categorical traits, swept across
    weight_fields x top_pcts.

    Convention: `top_pcts` values are the fraction of top-scoring variants to
    retain per weight field (e.g. 0.15 -> keep the top 15% of variants by
    score). Internally this is implemented as `threshold = 1.0 - top_pct`,
    i.e. a variant passes when `mt[w] >= threshold`. Matches
    run_all_models_batched_quant — the same top_pcts list means the same
    thing in both functions.

    Weights are continuous scores in [0, 1], not 0/1 pass/fail indicators.
    For a variant passing threshold with weight w_i, this test combines
    per-variant score statistics U_i = BETA_i / SE_i^2 (with Var(U_i) =
    1/SE_i^2) as:
        U_burden   = sum(w_i * U_i)            = sum(w_i * BETA_i / SE_i^2)
        Var(U_burden) = sum(w_i^2 * Var(U_i))  = sum(w_i^2 / SE_i^2)
        Z = U_burden / sqrt(Var(U_burden))
    The denominator must be weighted by w_i**2 (not a bare pass/fail count)
    whenever weights are continuous rather than binary.
    """

    mt = mt.repartition(17000)

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
            w_eff = hl.or_missing(hl.is_defined(mt[w]) & (mt[w] >= thresh), mt[w])

            agg_dict[f'sum_num_{tag}']  = hl.agg.sum(w_eff * mt.BETA / (mt.SE ** 2))
            agg_dict[f'sum_info_{tag}'] = hl.agg.sum((w_eff ** 2) / (mt.SE ** 2))
            agg_dict[f'n_var_{tag}']    = hl.agg.count_where(hl.is_defined(w_eff))

    gene_mt = mt.group_rows_by(mt.gene).aggregate(**agg_dict)

    gene_mt = gene_mt.checkpoint(
        'gs://aou_amc/data/scallion/genebass/burden_results/allmodels_burden_bin_tmp.mt',
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
        n_var = hl.agg.count_where(
            hl.any([w_dict[sf].any(lambda w: w > 0) for sf in score_fields])
        ),
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
        # _sigma2_y  = hl.agg.mean(ac * (mt.SE ** 2)),  # scalar; weight-independent
        _sigma2_y = hl.agg.mean(
            hl.or_missing(
                hl.any([w_dict[sf].any(lambda w: w > 0) for sf in score_fields]),
                ac * (mt.SE ** 2)
            )
        ),
        n_var      = hl.agg.count_where(
            hl.any([w_dict[sf].any(lambda w: w > 0) for sf in score_fields])
        ),
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




# ── Utils minor processing ──────────────────────────────────────────────
# def genebass_cleanup(mt, models):
#     cols_to_drop = [
#         'n_cases_defined',
#         'n_cases_both_sexes',
#         'n_cases_females',
#         'n_cases_males',
#         'description',
#         'description_more',
#         'coding_description'
#     ]
#     mt = mt.drop(*cols_to_drop)
#     row_fields = list(mt.row)
#     row_to_drop = [
#         f for f in row_fields
#         if (
#             f.endswith('_prob') or
#             f.endswith('_pred') or
#             (f.endswith('_pct') and f not in models) or
#             f in {
#                 'proteinmpnn_llr_neg', 'esm1b_neg', 'score_PAI3D', 'revel',
#                 'rasp_score', 'AM', 'MisFit_D', 'MisFit_S', 'polyphen_score',
#                 'cpt1_score', 'popEVE_neg', 'EVE', 'ESM_1v_neg', 'mpc',
#                 'cadd_score', 'gpn_msa_score_neg'
#             }
#         )
#     ]
#     mt = mt.drop(*row_to_drop)
#     return mt

def genebass_cleanup(ht, models):
    cols_to_drop = [
        'A1','A2', 'gene'
    ]
    ht = ht.drop(*cols_to_drop)
    row_fields = list(ht.row)
    row_to_drop = [
        f for f in row_fields
        if (
            f.endswith('_prob') or
            f.endswith('_pred') or
            (f.endswith('_pct') and f not in models) or
            f in {
                'proteinmpnn_llr_neg', 'esm1b_neg', 'score_PAI3D', 'revel',
                'rasp_score', 'AM', 'MisFit_D', 'MisFit_S', 'polyphen_score',
                'cpt1_score', 'popEVE_neg', 'EVE', 'ESM_1v_neg', 'mpc',
                'cadd_score', 'gpn_msa_score_neg'
            }
        )
    ]
    ht = ht.drop(*row_to_drop)
    return ht


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
    parser.add_argument('--use_flexrv', action='store_true',
                        help=(
                            'Use the FlexRV weight-grid approach '
                            '(192 combined score × MAF weights per score field). '
                            'If not set, the standard per-weight-field approach is used.'
                        ))
    parser.add_argument('--trait_type', choices=['qt', 'bin'], required=True,
                        help='Trait type to run: qt (quantitative/continuous) or bin (binary/categorical)')
    parser.add_argument('--test', action='store_true',
                        help='Run on test phenotype C50 only')
    parser.add_argument('--run_tmp', action='store_true',
                        help='Run temp code filter significant')
    return parser.parse_args()



def main(args):
    # hl.init(
    #     backend='batch',
    #     idempotent=True,
    #     tmp_dir='gs://aou_tmp/v8',
    #     log = "access_flexrv.log",
    #     app_name="Flexrv_in_genebass_phenos",
    #     worker_memory='10Gi',
    #     driver_memory='highmem',
    #     gcs_requester_pays_configuration="aou-neale-gwas"
    # )
    hl.init_batch(
        billing_project="all-by-aou",
        remote_tmpdir='gs://aou_tmp/v8',   # note: remote_tmpdir, not tmp_dir, for the batch backend
        worker_memory='10Gi',
        driver_memory='highmem',
        gcs_requester_pays_configuration="aou-neale-gwas",
    )

    # hl.current_backend().requester_pays_config = ('aou-neale-gwas', ['ukbb-exome-public'] )
    
    if args.run_tmp:
        run_temporary()

    if args.run_burden:
        var_path    = 'gs://ukbb-exome-public/500k/results/variant_results.mt'
        mt_genebass = hl.read_matrix_table(var_path)
        # ht_scallion = hl.read_table('gs://aou_amc/data/scallion/genebass/predictions/preds_by_chrom/all_chr_pct_preds.ht')
        ht_scallion = hl.import_table(
            'gs://aou_amc/scallion/benchmark/data/genebass_w_vsm_w_predictions_w_pct.tsv',
            impute=True,
        )
        ht_scallion = ht_scallion.annotate(
            locus=hl.parse_locus(ht_scallion.locus, reference_genome='GRCh38'),
            alleles=ht_scallion.alleles.replace(r'[\[\]"]', '').split(','),
        )
        ht_scallion = ht_scallion.key_by('locus', 'alleles')


        if args.use_flexrv:
            PRIMARY_SCORE_FIELDS = ['AM_pct', 'scallion_bin_Random-Forest_prob_pct']
            MAF_FIELD = 'AF.Cases'
            BATCH_SIZE = 150
            BASE_PATH = 'gs://aou_amc/data/scallion/burden/'

            print("[flexRV] Starting flexRV burden pipeline...")
            ht_scallion = genebass_cleanup(ht_scallion, PRIMARY_SCORE_FIELDS)

            # --- Configure trait type ---
            trait_config = {
                'bin': (
                    hl.literal(['icd10', 'categorical']).contains(mt_genebass.trait_type),
                    f'{BASE_PATH}/phenotypes_flexRV/binary_phenos.tsv',
                    run_flexrv_burden_bin,
                    'flexrv_burden_bin',
                ),
                'qt': (
                    mt_genebass.trait_type == 'continuous',
                    f'{BASE_PATH}/phenotypes_flexRV/qt_phenos.tsv',
                    run_flexrv_burden_quant,
                    'flexrv_burden_qt',
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
            print("[standard] Starting standard burden pipeline...")
            BASE_PATH = 'gs://aou_amc/data/scallion/genebass'
            
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
                    f'{BASE_PATH}/burden_results/tmp/standard_{test_pheno.lower()}_test.mt', overwrite=True
                )
                print(f"[standard] Test checkpoint written for {test_pheno}.")
            
            
            # print("[standard] Filtering entries (BETA/SE/AC validity)...")
            # --- Select run function and output path by trait type ---
            # if args.trait_type not in trait_config:
            #     raise ValueError(f"Invalid trait type: {args.trait_type}")

            if args.trait_type == 'bin':
                col_filter = hl.literal(["icd10", "categorical"]).contains(mt_genebass.trait_type) & (mt_genebass.modifier != "custom")
                run_fn = run_all_models_batched_bin
                out_path = f'{BASE_PATH}/burden_results_v2/tmp/burden_multibins.mt'
            elif args.trait_type == 'qt':
                col_filter = mt_genebass.trait_type == 'continuous'
                run_fn = run_all_models_batched_quant
                out_path = f'{BASE_PATH}/burden_results_v2/tmp/allmodels_burden_qt.mt'
            else:
                raise ValueError(f"Invalid trait type: {args.trait_type}")

            mt_filtered = mt_genebass.filter_cols(col_filter)

            mt_filtered = mt_filtered.filter_entries(
                hl.is_defined(mt_filtered.BETA) &
                hl.is_defined(mt_filtered.SE) & (mt_filtered.SE > 0) &
                hl.is_defined(mt_filtered.AC) &
                (mt_filtered.AC >= 1) & (mt_filtered.AC <= 100)
            )
            mt_filtered = mt_filtered.filter_rows(hl.agg.any(hl.is_defined(mt_filtered.BETA)))


            print(f"[standard] Running '{args.trait_type}' burden model...")
            # top_pcts = fraction of top-scoring variants retained per weight field
            # (e.g. 0.10 -> keep the top 10%). Same convention/list works for both
            # run_all_models_batched_bin and run_all_models_batched_quant.
            top_pcts = [0.10, 0.50, 0.75, 0.85]
            gene_mt = run_fn(mt_filtered, calibrated_scores, top_pcts=top_pcts)
            gene_mt = gene_mt.checkpoint(out_path, overwrite=True)
            print(f"[standard] Done — checkpoint written to {out_path}.")

            # mt_genebass = mt_genebass.filter_entries(
            #     hl.is_defined(mt_genebass.BETA) &
            #     hl.is_defined(mt_genebass.SE) & (mt_genebass.SE > 0) &
            #     hl.is_defined(mt_genebass.AC) &
            #     (mt_genebass.AC >= 1) & (mt_genebass.AC <= 100)
            # )
            # mt_genebass = mt_genebass.filter_rows(hl.agg.any(hl.is_defined(mt_genebass.BETA)))

            # # --- Select run function and output path by trait type ---
            # trait_config = {
            #     'bin': (
            #         mt_genebass.filter_cols(hl.literal(["icd10", "categorical"]).contains(mt_genebass.trait_type) & (mt_genebass.modifier != "custom")),
            #         run_all_models_batched_bin,
            #         # f'{BASE_PATH}/burden_results_v2/tmp/allmodels_burden_bin_weightFix_macFilter_top15bin.mt',
            #         f'{BASE_PATH}/burden_results_v2/tmp/allmodels_burden_bin_NOWEIGHT_macFilter_multiBins.mt',
            #     ),
            #     'qt': (
            #         mt_genebass.filter_cols(mt_genebass.trait_type == 'continuous'),
            #         run_all_models_batched_quant,
            #         f'{BASE_PATH}/burden_results_v2/tmp/allmodels_burden_qt.mt',
            #     ),
            # }

            # if args.trait_type not in trait_config:
            #     raise ValueError(f"Invalid trait type: {args.trait_type}")

            # mt_filtered, run_fn, out_path = trait_config[args.trait_type]
            # print(f"[standard] Running '{args.trait_type}' burden model...")
            # # top_pcts = fraction of top-scoring variants retained per weight field
            # # (e.g. 0.10 -> keep the top 10%). Same convention/list works for both
            # # run_all_models_batched_bin and run_all_models_batched_quant.
            # top_pcts = [0.10, 0.25, 0.50, 0.75, 0.85]
            # gene_mt = run_fn(mt_filtered, calibrated_scores, top_pcts=top_pcts)
            # gene_mt = gene_mt.checkpoint(out_path, overwrite=True)
            # print(f"[standard] Done — checkpoint written to {out_path}.")

    if args.summarize:
        # ── CONFIG ───────────────────────────────────────────────────────────────
        GENEBASS_PATH    = "gs://ukbb-exome-public/500k/results/results.mt"
        # BASE_PATH        = f"gs://aou_amc/data/scallion/genebass/burden_results_v2/allmodels_burden_{args.trait_type}_v2"
        BASE_PATH        = f"gs://aou_amc/data/scallion/genebass/burden_results_v2/tmp/allmodels_burden_{args.trait_type}_NOWEIGHT_macFilter_multiBins"
        WEIGHTED_RESULTS_PATH = f"{BASE_PATH}.mt"
        OUT_TSV          = f"{BASE_PATH}.tsv"
        OUT_HT           = f"{BASE_PATH}.ht"
        REKEY_GB_HT      = "gs://aou_amc/data/scallion/genebass/burden_results_v2/_tmp_gb_rekeyed.ht"
        REKEY_SC_HT      = "gs://aou_amc/data/scallion/genebass/burden_results_v2/_tmp_sc_rekeyed.ht"
        SIG_THRESHOLD    = 2.5e-6
        DROP_COLS = ['interval', 'markerIDs', 'description', 
                     'description_more', 'coding_description', 'category']
        
        # WEIGHTED_RESULTS_PATH = "gs://aou_amc/data/scallion/genebass/burden_results/allmodels_burden_qt_top15.mt"
        # OUT_TSV          = f"gs://aou_amc/data/scallion/genebass/burden_results_v2/allmodels_burden_bin.tsv"
        # OUT_HT           = f"gs://aou_amc/data/scallion/genebass/burden_results_v2/allmodels_burden_bin.ht"

        # ── 1. BUILD & CHECKPOINT gb_ht ──────────────────────────────────────────
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

        # ── 2. BUILD & CHECKPOINT sc_ht ──────────────────────────────────────────
        # sc_results = hl.read_matrix_table(BIN_RESULTS_PATH) # Read earlier
        def make_entry_sig(mt):
            return hl.any([
                (mt.entry[f] > 0) & (mt.entry[f] < SIG_THRESHOLD)
                for f in p_fields
            ])
        
        # p_fields = [f for f in sc_results.entry if f.startswith('p_scallion_')]
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

        # ── 3. JOIN ───────────────────────────────────────────────────────────────
        # Both sides are now tiny — the join and shuffle are cheap
        joined_ht = gb_ht.join(sc_ht, how='outer')

        # ── 4. WRITE OUTPUT ───────────────────────────────────────────────────────
        joined_ht = joined_ht.naive_coalesce(100).checkpoint(OUT_HT, overwrite=True)
        joined_ht.export(OUT_TSV)
        print(f"Significant rows written → {OUT_HT}")
        print(f"TSV exported             → {OUT_TSV}")


if __name__ == '__main__':
    main(parse_args())


