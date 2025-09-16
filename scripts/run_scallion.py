import hail as hl

import argparse
import hailtop.batch as hb
import os
import pandas as pd

def main(args):
    temp_dir = 'gs://aou_tmp'
    hl.init(app_name='Running scallion',
            idempotent=True,
            tmp_dir=temp_dir,
            default_reference = "GRCh38",
            gcs_requester_pays_configuration="aou-neale-gwas",
            log="/run_scallion.log")
    
    genebass_gene_path = "gs://ukbb-exome-public/500k/results/results.mt"
    save_out_ht = f'gs://aou_amc/scallion/data/pLoF_genebass_significant_nosparse.ht'
    phenos_cor_path = "gs://ukbb-exome-public/500k/qc/correlation_table_phenos_500k.ht"
    var_path = "gs://ukbb-exome-public/500k/results/variant_results.mt"
    results_prefix = "gs://aou_amc/scallion/results"


    if not hl.hadoop_exists(save_out_ht):
        from utils.processing import (
            filter_gene_matrix,
            get_significant_genes
        )
        
        gene_mt_clean = filter_gene_matrix(genebass_gene_path, phenos_cor_path, pval_threshold = 2.5e-6)
        gene_ht, plof_genes = get_significant_genes(gene_mt_clean, save_out_ht)
    else:
        plof_genes = pd.read_csv(f'{".".join(save_out_ht.rsplit(".", 1)[:-1]) + ".txt"}', header = None)
        plof_genes = plof_genes[0].values.tolist()
    
    if args.run_scallion:
        from utils.processing import (
            process_gene
        )
        
        backend = hb.ServiceBackend(
            billing_project="all-by-aou",
            remote_tmpdir="gs://aou_tmp/v8/",
        )

        b = hb.Batch(
            name="scallion run it",
            requester_pays_project="aou-neale-gwas",
            backend=backend)
        
        if args.test:
            plof_genes = plof_genes[:2]
            print(f"TEST MODE: Processing first 3 genes only: {plof_genes}")
        else:
            print(f"Processing {len(plof_genes)} genes")

            scan_results = hl.hadoop_ls('gs://aou_amc/scallion/results')
            genes_done = [
                os.path.basename(f['path']).replace('.csv', '')
                for f in scan_results
                if f['path'].endswith('.csv')
                ]
            plof_genes = [gene for gene in plof_genes if gene not in genes_done]
            

        gene_ht_path = save_out_ht
        
        for gene in plof_genes:
            j = b.new_python_job(f'run_scallion_{gene}', attributes={"gene": gene})
            j.image("hailgenetics/hail:0.2.133-py3.11")
            j.memory('highmem')
            j.cpu(8)
            j.env('PYSPARK_SUBMIT_ARGS', '--driver-memory 24g --executor-memory 24g pyspark-shell')
            try:
                j.call(process_gene,
                       gene,
                       gene_ht_path,
                       phenos_cor_path,
                       var_path,
                       results_prefix
                )
            except Exception as e:
                print(f"[!] Failed on {gene}: {e}")

        b.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run scallion.')
    
    parser.add_argument('--test', 
                        action='store_true', 
                        help='Use first 3 genes for testing')
    
    parser.add_argument('--run-scallion', 
                        action='store_true', 
                        help='Run scallion')
    
    args = parser.parse_args()
    main(args)
