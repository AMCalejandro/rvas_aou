import hail as hl
import pandas as pd

from utils.scallion import compute_scallion_scores

def filter_gene_matrix(
    gene_mt_path,
    phecor_ht_path,
    phecodes_keep=None,
    pval_threshold: float = 2.5e-6,
    checkpoint_path: str | None = None,
    n_partitions: int = 600,
):
    """
    Filter gene matrix for pLoF and significant burden associations,
    then clean (remove sparse rows/cols), and optionally checkpoint.

    Args:
        gene_mt: Hail MatrixTable
        phecodes_keep: list of phecodes to keep. If None, keep all.
        pval_threshold: significance threshold for Pvalue_Burden.
        checkpoint_path: if provided, checkpoint matrix at this path.
        n_partitions: number of partitions for checkpoint.

    Returns:
        Filtered and cleaned MatrixTable.
    """

    gene_mt = hl.read_matrix_table(gene_mt_path)

    if phecodes_keep is None:
        phecodes_keep = gene_mt.aggregate_cols(
            hl.agg.filter(
                ~gene_mt.phenocode.lower().contains('custom'),
                hl.agg.collect_as_set(gene_mt.phenocode)
            )
        )
        phecor_ht = hl.read_table(phecor_ht_path)
        phecodes_cor = phecor_ht.aggregate(hl.agg.collect_as_set(phecor_ht.i_pheno))
        phecodes_exclude = phecodes_keep - phecodes_cor
        phecodes_keep = list(phecodes_keep - phecodes_exclude)
        print(f"[i] Keeping {len(phecodes_keep)} phecodes after filtering out {len(phecodes_exclude)} custom phecodes not in correlation matrix.")

    mt = gene_mt.filter_cols(hl.literal(phecodes_keep).contains(gene_mt.phenocode))
    mt = mt.filter_rows(mt.annotation == "pLoF")
    mt = mt.filter_entries(mt.Pvalue_Burden < pval_threshold)

    mt = mt.filter_entries(hl.is_defined(mt.entry))
    mt = mt.filter_rows(hl.agg.count_where(hl.is_defined(mt.entry)) > 0)
    mt = mt.filter_cols(hl.agg.count_where(hl.is_defined(mt.entry)) > 0)

    if checkpoint_path is not None:
        mt = mt.naive_coalesce(n_partitions)
        mt = mt.checkpoint(checkpoint_path, overwrite=True)

    return mt


def get_significant_genes(mt, save_path: str, n_partitions: int = 100):
    """Extract non-sparse entries and return significant genes."""
    ht = mt.entries()
    ht = ht.key_by("gene_symbol", "phenocode").distinct().key_by()
    ht = ht.naive_coalesce(n_partitions)
    ht = ht.checkpoint(save_path, overwrite=True)

    gene_counts = ht.group_by(gene_symbol=ht.gene_symbol).aggregate(
        tot_count=hl.agg.count()
    )
    gene_counts = gene_counts.order_by(hl.desc("tot_count"))
    gene_counts = gene_counts.filter(gene_counts.tot_count > 1)

    save_out_genes_txt = '.'.join(save_path.rsplit('.', 1)[:-1]) + '.txt'
    gene_counts.select("gene_symbol").export(save_out_genes_txt, header=False, delimiter=" ")
    plof_genes = list(gene_counts.aggregate(hl.agg.collect(gene_counts.gene_symbol)))
    
    return ht, plof_genes


def get_gene_phenotype_correlations(ht, phenos_cor_path: str, gene: str):
    """Get phenotype correlations for a given gene."""
    phenos_cor = hl.read_table(phenos_cor_path)

    b_plof = ht.filter(ht.gene_symbol == gene).key_by().select(
        "phenocode", "coding", "BETA_Burden"
    )

    b_plof_phecodes = b_plof.aggregate(hl.agg.collect_as_set(b_plof.phenocode))

    phenos_cor = phenos_cor.filter(
        hl.set(b_plof_phecodes).contains(phenos_cor.i_pheno)
        & hl.set(b_plof_phecodes).contains(phenos_cor.j_pheno)
    )

    # phenos_cor_phecodes_set = phenos_cor.aggregate(
    #     hl.agg.collect_as_set(phenos_cor.j_pheno)
    # )
    # missing_phecodes = b_plof_phecodes - phenos_cor_phecodes_set
    # if missing_phecodes:
    #     print(f'There are missing phenotypes in the correlation matrix for {gene}')
    #     b_plof = b_plof.filter(~hl.set(missing_phecodes).contains(b_plof.phenocode))
    
    pheno_coding_keys = b_plof.aggregate(
        hl.agg.collect_as_set(
            hl.struct(phenocode=b_plof.phenocode, coding=b_plof.coding)
        )
    )

    blof_pd = b_plof.to_pandas()
    pheno_corr_pd = phenos_cor.to_pandas()
    pivot_matrix = pheno_corr_pd.pivot_table(
        index="i_pheno", columns="j_pheno", values="entry", fill_value=0
    )

    return pheno_coding_keys, blof_pd, pivot_matrix


def filter_variant_matrix(var_path: str, gene: str, pheno_coding_keys, save_path: str):
    """Filter variant-level matrix for given gene and pheno_coding_keys."""
    var_mt = hl.read_matrix_table(var_path).key_cols_by()

    var_mt_filtered = var_mt.filter_cols(
        hl.literal(pheno_coding_keys).contains(
            hl.struct(phenocode=var_mt.phenocode, coding=var_mt.coding)
        )
    )

    missense_annots = ["missense"]
    var_mt_filtered = var_mt_filtered.filter_rows(
        (var_mt_filtered.gene == gene)
        & (hl.literal(missense_annots).contains(var_mt_filtered.annotation))
    )

    var_mt_filtered_ht = var_mt_filtered.entries()
    var_mt_filtered_ht = var_mt_filtered_ht.naive_coalesce(250)
    var_mt_filtered_ht = var_mt_filtered_ht.checkpoint(save_path, overwrite=True)

    var_missense = var_mt_filtered_ht.select(
        "markerID", "phenocode", "coding", "BETA", "SE"
    )

    var_missense = var_missense.to_pandas().drop(columns=["locus", "alleles"])
    
    return var_mt_filtered_ht, var_missense


def process_gene(
    gene: str,
    gene_ht_path,
    phenos_cor_path: str,
    var_path: str,
    results_prefix: str,
):
    """Run full pipeline for a single gene."""

    temp_dir = 'gs://aou_tmp'
    hl.init(
        master='local[32]',
        tmp_dir=temp_dir,
        gcs_requester_pays_configuration='aou-neale-gwas',
        worker_memory="highmem",
        worker_cores=8,
        default_reference="GRCh38",
    )
    
    pheno_coding_keys, blof_pd, pivot_matrix = get_gene_phenotype_correlations(
        gene_ht_path=gene_ht_path,
        phenos_cor_path=phenos_cor_path,
        gene=gene,
    )
    blof_pd = blof_pd.sort_values("phenocode")

    save_path = f"{results_prefix}/tmp/{gene}_variants.mt"
    _, var_missense_pd = filter_variant_matrix(
        var_path, gene, pheno_coding_keys, save_path
    )
    var_missense_pd = var_missense_pd.sort_values("phenocode")

    assert (
        blof_pd.shape[0]
        == len(var_missense_pd.phenocode.unique())
        == pivot_matrix.shape[0]
    )
    
    missense_beta_pd = (
        var_missense_pd
        .pivot(index='markerID', columns='phenocode', values='BETA')
        .astype(float)
    )
    missense_se_pd = (
        var_missense_pd
        .pivot(index='markerID', columns='phenocode', values='SE')
        .astype(float)
    )

    df_scores = compute_scallion_scores(
        gene = gene,
        beta_lof=blof_pd["BETA_Burden"].values,
        P=pivot_matrix,
        missense_betas=missense_beta_pd,
        missense_ses=missense_se_pd,
        mask_missing=True,
        min_traits=2,
    )

    out_path = f"{results_prefix}/{gene}.csv"
    df_scores_ht = hl.Table.from_pandas(df_scores)
    try:
        df_scores_ht.export(out_path)
    except:
        out_path = f"{results_prefix}/{gene}_retry.ht"
        df_scores_ht.export(out_path)
    print(f"[✔] Saved results for {gene} → {out_path}")