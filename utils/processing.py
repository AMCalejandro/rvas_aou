import os
import hail as hl

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
                ~gene_mt.modifier.lower().contains('custom'),
                hl.agg.collect_as_set(gene_mt.phenocode)
            )
        )
        phecor_ht = hl.read_table(phecor_ht_path)
        phecodes_cor = phecor_ht.aggregate(hl.agg.collect_as_set(phecor_ht.i_pheno))
        phecodes_exclude = phecodes_keep - phecodes_cor
        phecodes_keep = list(phecodes_keep - phecodes_exclude)
        print(f"[i] Keeping {len(phecodes_keep)} phecodes after filtering out {len(phecodes_exclude)} custom phecodes not in correlation matrix.")

    gene_mt = gene_mt.filter_cols(hl.literal(phecodes_keep).contains(gene_mt.phenocode))
    gene_mt = gene_mt.filter_rows(gene_mt.annotation == "pLoF")
    gene_mt = gene_mt.filter_entries(
        hl.any(lambda p: p < pval_threshold,
            [gene_mt.Pvalue_Burden])
    )
    gene_mt = gene_mt.filter_rows(hl.agg.count_where(hl.is_defined(gene_mt.entry)) > 0)
    gene_mt = gene_mt.filter_cols(hl.agg.count_where(hl.is_defined(gene_mt.entry)) > 0)

    if checkpoint_path is not None:
        gene_mt = gene_mt.naive_coalesce(n_partitions)
        gene_mt = gene_mt.checkpoint(checkpoint_path, overwrite=True)

    return gene_mt

def get_significant_genes(mt, 
                          save_path: str, 
                          n_partitions: int = 100, 
                          mode: str = 'multi'):
    """Extract non-sparse entries and return significant genes."""
    ht = mt.entries()
    ht = ht.key_by('gene_symbol','phenocode','coding').distinct().key_by()
    ht = ht.naive_coalesce(n_partitions)
    ht = ht.checkpoint(save_path, overwrite=True)

    gene_counts = ht.group_by(gene_symbol=ht.gene_symbol).aggregate(
        tot_count=hl.agg.count()
    )
    gene_counts = gene_counts.order_by(hl.desc("tot_count"))
    if mode == 'multi':
        gene_counts = gene_counts.filter(gene_counts.tot_count > 1)

    save_out_genes_txt = '.'.join(save_path.rsplit('.', 1)[:-1]) + f'_{mode}.txt'
    gene_counts.select("gene_symbol").export(save_out_genes_txt, header=False, delimiter=" ")
    plof_genes = list(gene_counts.aggregate(hl.agg.collect(gene_counts.gene_symbol)))
    
    return ht, plof_genes

def get_remaining_genes(plof_genes, results_prefix, overwrite=False):
    """
    Filter `plof_genes` down to those that don't already have a results CSV
    in `results_prefix`, unless `overwrite` is True.

    Parameters
    ----------
    plof_genes : list[str]
        Candidate genes to run.
    results_prefix : str
        GCS path prefix where per-gene outputs live, e.g.
        'gs://aou_amc/scallion/results_final/test/'. Each gene's result is
        expected at f"{results_prefix}{gene}.csv".
    overwrite : bool
        If True, skip the existence check and return the full gene list.

    Returns
    -------
    list[str]
        Genes still needing to be run.
    """
    if overwrite:
        print(f"Overwrite=True: running all {len(plof_genes)} genes.")
        return plof_genes

    if not hl.hadoop_exists(results_prefix):
        print(f"No existing results at {results_prefix}. Running all {len(plof_genes)} genes.")
        return plof_genes

    existing_files = hl.hadoop_ls(results_prefix)
    existing_genes = {
        os.path.basename(f['path'])[:-len('.csv')]
        for f in existing_files
        if f['path'].endswith('.csv')
    }

    remaining_genes = [g for g in plof_genes if g not in existing_genes]

    n_done = len(plof_genes) - len(remaining_genes)
    print(
        f"Found {n_done} completed genes out of {len(plof_genes)} in {results_prefix}. "
        f"{len(remaining_genes)} remaining."
    )

    return remaining_genes