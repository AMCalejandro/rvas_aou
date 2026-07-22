import hail as hl
import pandas as pd
from gnomad.resources.grch38.reference_data import clinvar


# ============================================================================
# Shared constants
# ============================================================================
LOF_CONSEQUENCES = {
    "stop_gained",
    "frameshift_variant",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "start_lost",
    "stop_lost",
    "transcript_ablation",
}
MISSENSE_CONSEQUENCES = {"missense_variant"}
SYNONYMOUS_CONSEQUENCES = {"synonymous_variant"}
SPLICE_UNCERTAIN_CONSEQUENCES = {
    "splice_region_variant",
    "splice_donor_region_variant",
    "splice_donor_5th_base_variant",
    "splice_polypyrimidine_tract_variant",
}

KEEP_FIELDS = [
    "locus", "alleles", "rsid",
    "in_scallion_genes", "gene_names_array", "GENEINFO",
    "category",
    "most_severe_consequence",
    "variant_class",
    "transcript_consequences",
    "CLNSIG_flat", "CLNSIG_primary",
    "CLNSIG_mapped", "CLNSIG_is_simple",
    "CLNREVSTAT_flat", "CLNREVSTAT_numeric",
]

KEEP_CLNSIG = {"Pathogenic", "Likely_pathogenic", "Benign", "Likely_benign"}
DELETERIOUS = hl.literal({"Pathogenic", "Likely_pathogenic"})

WEIGHTS = {"Pathogenic": 1.0, "Likely_pathogenic": 0.9, "Benign": 0.0, "Likely_benign": 0.1}

DEFAULT_SC_GENES_ALL_PATH = (
    "gs://aou_amc/scallion/dev/pLoF_genebass_significant_nosparse_single.txt"
)
DEFAULT_SC_GENES_MULTI_PATH = (
    "gs://aou_amc/scallion/dev/pLoF_genebass_significant_nosparse_multi.txt"
)
DEFAULT_OUTPUT_PATH = "gs://aou_amc/data/utils/clinvar/clinvar_all.ht"
DEFAULT_HIGH_ENRICHMENT_OUTPUT = "gs://aou_amc/data/utils/clinvar/high_enrichment.tsv"
DEFAULT_LOW_ENRICHMENT_OUTPUT = "gs://aou_amc/data/utils/clinvar/low_enrichment.tsv"

MIN_MISSENSE_VARIANTS = 10   # floor on classified missense variants/gene
N_GENES = 20
Z = 1.96                     # ~95% Wilson interval


# ============================================================================
# Part 1: build - construct the filtered/annotated ClinVar Hail Table
# ============================================================================
def _load_gene_list(path):
    """Read a single-column, header-less file of gene symbols into a list."""
    return pd.read_csv(path, header=None)[0].tolist()

def _map_clnsig(clnsig_primary):
    """Collapse a raw (lowercased) CLNSIG value into a small set of categories."""
    return (
        hl.case()
        .when(clnsig_primary.contains("conflicting"), "Conflicting_classifications")
        .when(clnsig_primary == "pathogenic", "Pathogenic")
        .when(clnsig_primary == "likely_pathogenic", "Likely_pathogenic")
        .when(clnsig_primary == "benign", "Benign")
        .when(clnsig_primary == "likely_benign", "Likely_benign")
        .when(clnsig_primary.contains("uncertain"), "Uncertain_significance")
        .when(clnsig_primary.contains("risk"), "Risk_factor")
        .when(clnsig_primary.contains("drug_response"), "Drug_response")
        .when(clnsig_primary.contains("protective"), "Protective")
        .when(clnsig_primary.contains("association"), "Association")
        .when(clnsig_primary.contains("affects"), "Affects")
        .when(
            clnsig_primary.contains("not_provided")
            | clnsig_primary.contains("no_classification"),
            "Not_classified",
        )
        .default("Other")
    )

def _map_clnrevstat_numeric(clnrevstat_flat):
    """Map a raw (lowercased) CLNREVSTAT string to a 0-6 confidence score."""
    return (
        hl.case()
        .when(hl.is_missing(clnrevstat_flat), 0)
        .when(clnrevstat_flat.contains("no_classification"), 0)
        .when(clnrevstat_flat.contains("no_assertion"), 1)
        .when(clnrevstat_flat.contains("single_submitter"), 2)
        .when(clnrevstat_flat.contains("conflicting"), 3)
        .when(clnrevstat_flat.contains("multiple_submitters"), 4)
        .when(clnrevstat_flat.contains("expert_panel"), 5)
        .when(clnrevstat_flat.contains("practice_guideline"), 6)
        .default(0)
    )

def _map_category(most_severe_consequence):
    """Bucket a VEP most_severe_consequence into a coarse functional category."""
    lof = hl.literal(LOF_CONSEQUENCES)
    missense = hl.literal(MISSENSE_CONSEQUENCES)
    synonymous = hl.literal(SYNONYMOUS_CONSEQUENCES)
    splice_uncertain = hl.literal(SPLICE_UNCERTAIN_CONSEQUENCES)
    return (
        hl.case()
        .when(lof.contains(most_severe_consequence), "LoF")
        .when(missense.contains(most_severe_consequence), "Missense")
        .when(synonymous.contains(most_severe_consequence), "Synonymous")
        .when(splice_uncertain.contains(most_severe_consequence), "Splice_uncertain")
        .default("Other")
    )

def build_clinvar_ht(
    sc_genes_all_path=DEFAULT_SC_GENES_ALL_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
    n_partitions=100,
    overwrite=True,
):
    """
    Build a filtered, annotated ClinVar Hail Table restricted to
    Pathogenic / Likely_pathogenic / Benign / Likely_benign variants,
    flagging whether each variant falls in a given ("scallion") gene set.

    Parameters
    ----------
    sc_genes_all_path : str
        Path to a single-column, header-less file of gene symbols used to
        flag `in_scallion_genes`.
    output_path : str
        Where to checkpoint the resulting Hail Table.
    n_partitions : int
        Number of partitions to coalesce to before checkpointing.
    overwrite : bool
        Whether to overwrite an existing table at `output_path`.

    Returns
    -------
    hail.Table
        The checkpointed, filtered ClinVar table.
    """
    sc_genes_all = _load_gene_list(sc_genes_all_path)
    sc_genes_set = hl.literal(set(sc_genes_all))

    ht = clinvar.ht()
    ht = ht.annotate(
        GENEINFO=ht.info.GENEINFO,
        CLNSIG=ht.info.CLNSIG,
        CLNREVSTAT=ht.info.CLNREVSTAT,
        variant_class=ht.vep.variant_class,
        most_severe_consequence=ht.vep.most_severe_consequence,
        transcript_consequences=ht.vep.transcript_consequences,
    ).drop("info", "vep")

    # GENEINFO looks like "GENE1:1234|GENE2:5678" -> pull out just the gene
    # symbols once, then reuse for both the rejoined GENEINFO string and the
    # gene_names_array field.
    gene_names_array = ht.GENEINFO.split(r"\|").map(lambda x: x.split(":")[0])
    ht = ht.annotate(
        GENEINFO=hl.delimit(gene_names_array, "|"),
        gene_names_array=gene_names_array,
    )
    ht = ht.annotate(
        in_scallion_genes=ht.gene_names_array.any(lambda g: sc_genes_set.contains(g))
    )

    # Flatten CLNSIG (array<str>) to a single lowercase string, then take the
    # first "|"- and "/"-delimited token as the primary classification.
    ht = ht.annotate(
        CLNSIG_flat=hl.or_missing(
            hl.is_defined(ht.CLNSIG),
            hl.delimit(ht.CLNSIG, delimiter="").lower(),
        )
    )

    ht = ht.annotate(
        CLNSIG_primary=hl.or_missing(
            hl.is_defined(ht.CLNSIG_flat),
            ht.CLNSIG_flat.split(r"\|")[0].split(r"/")[0],
        )
    )

    ht = ht.annotate(
        CLNREVSTAT_flat=hl.or_missing(
            hl.is_defined(ht.CLNREVSTAT),
            hl.delimit(ht.CLNREVSTAT, delimiter="").lower(),
        )
    )

    ht = ht.annotate(
        CLNSIG_mapped=hl.if_else(
            hl.is_missing(ht.CLNSIG_primary),
            "Not_classified",
            _map_clnsig(ht.CLNSIG_primary),
        ),
        CLNSIG_is_simple=hl.if_else(
            hl.is_missing(ht.CLNSIG_flat),
            False,
            (
                (~ht.CLNSIG_flat.contains(r"\|") & ~ht.CLNSIG_flat.contains("/"))
                | (ht.CLNSIG_flat == "benign/likely_benign")
                | (ht.CLNSIG_flat == "pathogenic/likely_pathogenic")
            ),
        ),
        CLNREVSTAT_numeric=_map_clnrevstat_numeric(ht.CLNREVSTAT_flat),
        category=_map_category(ht.most_severe_consequence),
    )

    ht = ht.key_by().select(*KEEP_FIELDS)

    keep_clnsig = hl.literal(KEEP_CLNSIG)
    ht = ht.filter(keep_clnsig.contains(ht.CLNSIG_mapped))

    # Filter out low quality variants
    ht = ht.filter(
        (ht.CLNREVSTAT_numeric > 3) |
        (ht.CLNREVSTAT_numeric == 2)
    )

    ht = ht.naive_coalesce(n_partitions).checkpoint(output_path, overwrite=overwrite)

    return ht


# ============================================================================
# Part 2: report - pathogenic priors per functional category
# ============================================================================
def compute_prior(label_counts):
    """Empirical pathogenic prior: (P + LP) / (P + LP + B + LB)."""
    p = label_counts.get("Pathogenic", 0)
    lp = label_counts.get("Likely_pathogenic", 0)
    b = label_counts.get("Benign", 0)
    lb = label_counts.get("Likely_benign", 0)
    total = p + lp + b + lb
    return (p + lp) / total if total else None

def weighted_prior(label_counts, weights=WEIGHTS):
    """Severity-weighted pathogenic prior using `weights`."""
    num = sum(weights[k] * v for k, v in label_counts.items() if k in weights)
    den = sum(v for k, v in label_counts.items() if k in weights)
    return num / den if den else None

def report_clinvar_priors(input_path=DEFAULT_OUTPUT_PATH, output_csv=None):
    """
    Read the ClinVar table built by `build_clinvar_ht` and print, per
    functional category, the raw CLNSIG_mapped counts plus an empirical
    and a severity-weighted pathogenic prior.

    Parameters
    ----------
    input_path : str
        Path to the checkpointed Hail Table to report on.
    output_csv : str or None
        If given, also write a tidy CSV of per-category priors here.

    Returns
    -------
    dict
        {category: {"empirical_prior": ..., "weighted_prior": ..., "counts": ...}}
    """
    ht = hl.read_table(input_path)
    print(ht.count())

    ht = ht.filter(hl.is_defined(ht.CLNSIG_mapped) & hl.is_defined(ht.category))
    counts = ht.aggregate(
        hl.agg.group_by(ht.category, hl.agg.counter(ht.CLNSIG_mapped))
    )

    for cat, label_counts in counts.items():
        print(cat, round(compute_prior(label_counts), 4), label_counts)

    print("\n")

    for cat, label_counts in counts.items():
        print(cat, round(weighted_prior(label_counts), 4), label_counts)

    results = {
        cat: {
            "empirical_prior": compute_prior(label_counts),
            "weighted_prior": weighted_prior(label_counts),
            "counts": dict(label_counts),
        }
        for cat, label_counts in counts.items()
    }

    if output_csv:
        rows = [
            {
                "category": cat,
                "empirical_prior": vals["empirical_prior"],
                "weighted_prior": vals["weighted_prior"],
                **vals["counts"],
            }
            for cat, vals in results.items()
        ]
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"Wrote per-category priors to {output_csv}")

    return results


# ============================================================================
# Part 3: select - high/low enrichment genes within scallion gene sets
# ============================================================================
def _load_scallion_gene_labels(all_path=DEFAULT_SC_GENES_ALL_PATH,
                                multi_path=DEFAULT_SC_GENES_MULTI_PATH):
    """
    Returns a hl.dict mapping gene -> "single" or "multi", built from the
    scallion gene list files. "single" genes are defined as
    (all_genes - multi_genes); everything in multi_genes is labeled "multi".
    """
    all_genes = pd.read_csv(all_path, header=None)
    multi_genes = pd.read_csv(multi_path, header=None)

    multi_gene_set = set(multi_genes[0].tolist())
    single_gene_set = set(all_genes[0].tolist()) - multi_gene_set

    label_map = {g: "multi" for g in multi_gene_set}
    label_map.update({g: "single" for g in single_gene_set})

    return hl.literal(label_map)

def top_n_per_group(ht, group_field, value_fields, order_expr, n, descending=True):
    ordering = -order_expr if descending else order_expr
    grouped = (
        ht.group_by(ht[group_field])
        .aggregate(
            rows=hl.agg.take(
                ht.row.select(*value_fields),
                n,
                ordering=ordering,
            )
        )
    )
    grouped = grouped.explode("rows")
    grouped = grouped.select(**grouped.rows)  # no group_field here
    return grouped

def select_enriched_genes(
    ht,
    sc_genes_all_path=DEFAULT_SC_GENES_ALL_PATH,
    sc_genes_multi_path=DEFAULT_SC_GENES_MULTI_PATH,
    min_missense_variants=MIN_MISSENSE_VARIANTS,
    n_genes=N_GENES,
    z=Z,
):
    """
    Select genes with high vs. low enrichment of deleterious missense
    variants from a ClinVar-derived Hail Table, restricted to genes flagged
    as `in_scallion_genes`.

    Assumes `ht` has already been through the QC steps in `build_clinvar_ht`:
      - CLNSIG_mapped restricted to {Pathogenic, Likely_pathogenic, Benign, Likely_benign}
      - CLNREVSTAT_numeric > 3 OR == 2  (review-status / star filter)

    Because those two filters mean every surviving variant is unambiguously
    classified as deleterious or benign, "enrichment of deleterious missense
    variants" for a gene reduces to:

        p_hat(gene) = (# deleterious missense variants in gene)
                      / (# classified missense variants in gene)

    The number of classified missense variants observed per gene is the
    proxy used for "gene size" (there's no CDS-length field in the schema),
    but that also means genes with very few missense variants can land at
    p_hat = 0 or 1 purely by chance. Sorting on raw p_hat would just surface
    a pile of 1-2 variant genes at both extremes.

    This is handled with a Wilson score interval, which shrinks noisy,
    low-n estimates toward the middle:
      - ranking by the interval's LOWER bound, descending, picks genes that
        are confidently high (needs both a high p_hat AND enough variants
        to trust it)
      - ranking by the interval's UPPER bound, ascending, picks genes that
        are confidently low

    A hard `min_missense_variants` floor is also applied as an extra sanity
    check on top of the Wilson shrinkage.

    Each gene in the result tables is labeled "single" or "multi" depending
    on whether it appears in the scallion single-gene list or the scallion
    multi-gene list (the single-gene list is defined as the set difference:
    all scallion genes minus multi genes).

    Returns
    -------
    (hail.Table, hail.Table)
        (high_enrichment, low_enrichment), each with columns:
        gene, n_missense, n_deleterious, p_hat, wilson_lower/wilson_upper.
    """
    # ---- 0. build gene -> single/multi label map ---------------------------
    gene_label_map = _load_scallion_gene_labels(sc_genes_all_path, sc_genes_multi_path)

    # ---- 1. restrict to scallion genes ------------------------------------
    ht = ht.filter(ht.in_scallion_genes)

    # a variant can map to >1 gene symbol; explode so each gene is counted
    # against its own totals
    ht = ht.filter(hl.is_defined(ht.gene_names_array) & (hl.len(ht.gene_names_array) > 0))
    ht = ht.annotate(gene=ht.gene_names_array)
    ht = ht.explode(ht.gene)

    # ---- 2. restrict to missense, flag deleterious -------------------------
    missense_ht = ht.filter(ht.category == "Missense")
    missense_ht = missense_ht.annotate(
        is_deleterious=DELETERIOUS.contains(missense_ht.CLNSIG_mapped)
    )

    # ---- 3. per-gene counts -------------------------------------------------
    gene_ht = missense_ht.group_by(missense_ht.gene).aggregate(
        n_missense=hl.agg.count(),
        n_deleterious=hl.agg.count_where(missense_ht.is_deleterious),
    )
    gene_ht = gene_ht.annotate(
        n_benign=gene_ht.n_missense - gene_ht.n_deleterious,
        p_hat=gene_ht.n_deleterious / gene_ht.n_missense,
    )

    # ---- 3b. label each gene as scallion single vs. multi ------------------
    # get_or_else guards against genes that ended up in gene_ht (e.g. via
    # gene_names_array synonyms) but aren't an exact key match in either list
    gene_ht = gene_ht.annotate(
        scallion_label=gene_label_map.get(gene_ht.gene, "unlabeled")
    )
    gene_ht = gene_ht.filter(gene_ht.scallion_label != "unlabeled")

    # ---- 4. Wilson score interval (accounts for variable variant counts) --
    n = gene_ht.n_missense
    p = gene_ht.p_hat
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * hl.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))

    gene_ht = gene_ht.annotate(
        wilson_lower=center - margin,
        wilson_upper=center + margin,
    )

    gene_ht = gene_ht.checkpoint(hl.utils.new_temp_file("gene_missense_stats", "ht"))

    # diagnostic: look at this before finalizing min_missense_variants
    print(gene_ht.n_missense.summarize())

    # ---- 5. apply evidence floor, take the extremes ------------------------
    evidenced_ht = gene_ht.filter(gene_ht.n_missense >= min_missense_variants)

    value_fields = ["gene", "n_missense", "n_deleterious", "p_hat"]

    high_enrichment = top_n_per_group(
        evidenced_ht,
        "scallion_label",
        value_fields + ["wilson_lower"],
        evidenced_ht.wilson_lower,
        n_genes,
        descending=True,
    )

    low_enrichment = top_n_per_group(
        evidenced_ht,
        "scallion_label",
        value_fields + ["wilson_upper"],
        evidenced_ht.wilson_upper,
        n_genes,
        descending=False,
    )

    return high_enrichment, low_enrichment