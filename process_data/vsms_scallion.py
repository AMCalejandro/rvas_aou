import hail as hl
import pyarrow.dataset as ds
import pandas as pd
import pickle
import argparse



SCALLION_COLS = [
    'chrom', 'pos', 'ref', 'alt', 'mean_AC', 'gene_symbol', 
    'scallion_llr', 'scallion_prob_lof_signed', 'scallion_prob_mixture'
]

VSMS_COLS = [
    'chrom', 'pos', 'ref', 'alt', 'ensg',
    'AM', 'mcap', 'esm1b', 'gmvp', 'phylop', 'sift', 'cadd',
    'cpt', 'gpn_msa', 'ESM_1v', 'EVE', 'popEVE', 'PAI3D',
    'MisFit_D', 'MisFit_S', 'mpc', 'polyphen'
]

missing_ids = {
    'AC022414.1': 'ENSG00000284762',
    'AC069368.1': 'ENSG00000249240',
    'AC073896.1': 'ENSG00000144785',
    'AL136295.3': 'ENSG00000259371',
    'ANGPTL2': 'ENSG00000136859',
    'ANGPTL3': 'ENSG00000132855',
    'APBB3': 'ENSG00000113108',
    'APOA1': 'ENSG00000118137',
    'APOA5': 'ENSG00000110243',
    'ATP5MGL': 'ENSG00000249222', 
    'CAPZA3': 'ENSG00000177938',
    'CCDC157': 'ENSG00000187860', 
    'CCNA2': 'ENSG00000145386',
    'CHST9': 'ENSG00000154080', 
    'CHTF18': 'ENSG00000127586',
    'CLIC3': 'ENSG00000169583', 
    'CTRL': 'ENSG00000141086', 
    'DND1': 'ENSG00000256453',
    'E2F4': 'ENSG00000205250',
    'EIF6': 'ENSG00000242372', 
    'GP1BA': 'ENSG00000185245', 
    'GPR151': 'ENSG00000173250', 
    'GPT': 'ENSG00000167701',
    'HBA1': 'ENSG00000206172', 
    'HSF4': 'ENSG00000102878', 
    'IFRD2': 'ENSG00000214706',
    'INHBE': 'ENSG00000139269',
    'KLF1': 'ENSG00000105610', 
    'KLK11': 'ENSG00000167757',
    'LRRC18': 'ENSG00000165383',
    'NDUFAF3': 'ENSG00000178057.',
    'NEURL2': 'ENSG00000124257',
    'NIT1': 'ENSG00000158793',
    'PPP1R3D': 'ENSG00000132825',
    'PSMB11': 'ENSG00000222028',
    'RBM12': 'ENSG00000244462',
    'SPHK1': 'ENSG00000176170',
    'STK16': 'ENSG00000115661',
    'THAP11': 'ENSG00000168286',
    'TMEM102': 'ENSG00000181284',
    'TNFSF13': 'ENSG00000161955',
    'UGT1A1': 'ENSG00000241635',
    'UGT1A4': 'ENSG00000244474',
    'UGT1A5': 'ENSG00000288705',
    'UGT1A7': 'ENSG00000244122',
    'UGT1A9': 'ENSG00000241119',
    'WDR6': 'ENSG00000178252'
}



def parse_args():
    parser = argparse.ArgumentParser(description="Build scallion paths from a legacy run ID")
    parser.add_argument(
        '--scallion_prefix',
        type=str,
        required=True,
        help="Legacy identifier used to locate the scallion results directory (e.g. 'run_2024_v1')"
    )
    parser.add_argument(
        '--merge_type',
        type=str,
        required=True,
        help="Type of merge operation to perform. Currently only 'vsm_scallion' is supported."
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help="If set, re-run and overwrite outputs even if they already exist."
    )
    return parser.parse_args()


def main(args):
    
    scallion_prefix = args.scallion_prefix
    merge_type = args.merge_type
    
    hl.init(
        app_name=f'Merge data {merge_type}',
        idempotent=True,
        tmp_dir='gs://aou_tmp',
        default_reference="GRCh38",
        gcs_requester_pays_configuration="aou-neale-gwas",
        log="/run_scallion.log",
    )

    if merge_type == 'vsm_scallion':
        print(f"merge_type '{merge_type}' ")
        ensg_genesymbol_map     = "gs://aou_amc/scallion/utils/genesymbol_to_ensg.pkl"
        scallion_path           = f'gs://aou_amc/scallion/results_final/{scallion_prefix}/scallion_concatenated_qc.tsv'
        data_processed_tmp_path = f'gs://aou_amc/scallion/training/input_data/scallion_w_vsm_{scallion_prefix}_tmp.tsv.gz'
        data_processed_path     = f'gs://aou_amc/scallion/training/input_data/scallion_w_vsm_{scallion_prefix}.tsv.gz'
        vsms_inner              = 'gs://grohlicek/genetics_gym_vsm_all_content/full_analysis_tables/gene_aggregated/variant_scores_all_outer_ensg_stats_eval.parquet'
        
        print(f"scallion_path: {scallion_path}")
        print(f"data_processed_path: {data_processed_path}")

        if not hl.hadoop_exists(ensg_genesymbol_map):
            path_genebass_vep = 'gs://ukbb-exome-public/500k/results/vep.ht/'
            ht = hl.read_table(path_genebass_vep)
            ht = ht.explode(ht.vep.transcript_consequences)

            ht_pairs = ht.select(
                gene_symbol=ht.vep.transcript_consequences.gene_symbol,
                gene_id=ht.vep.transcript_consequences.gene_id
            )
            ht_pairs = ht_pairs.distinct()
            pairs = ht_pairs.select(ht_pairs.gene_symbol, ht_pairs.gene_id).collect()

            gene_symbol_to_ensg = {row.gene_symbol: row.gene_id for row in pairs}

            with hl.hadoop_open('gs://aou_amc/scallion/utils/genesymbol_to_ensg.pkl', 'wb') as f:
                pickle.dump(gene_symbol_to_ensg, f)
        else:
            if not hl.hadoop_exists(data_processed_tmp_path) or args.overwrite:
                with hl.hadoop_open('gs://aou_amc/scallion/utils/genesymbol_to_ensg.pkl', 'rb') as f:
                    gene_symbol_to_ensg = pickle.load(f)

                scallion = pd.read_csv(scallion_path, sep = '\t')
                pattern = r'(?P<chrom>chr\w+):(?P<pos>\d+)_(?P<ref>[ACGT]+)/(?P<alt>[ACGT]+)'
                scallion[['chrom', 'pos', 'ref', 'alt']] = scallion['markerID'].str.extract(pattern)
                scallion['pos'] = scallion['pos'].astype(int)
                scallion = scallion[SCALLION_COLS]

                gene_symbol_idx = scallion.columns.get_loc('gene_symbol')
                ensembl_ids = scallion['gene_symbol'].map(gene_symbol_to_ensg)
                scallion.insert(gene_symbol_idx + 1, 'ensembl_id', ensembl_ids)
                mask = scallion['ensembl_id'].isna()
                scallion.loc[mask, 'ensembl_id'] = scallion.loc[mask, 'gene_symbol'].map(missing_ids)

                scallion.to_csv(data_processed_tmp_path, sep = '\t', index=False)
            else:
                scallion = pd.read_csv(data_processed_tmp_path, sep = '\t')

        dataset = ds.dataset(
            vsms_inner,
            format="parquet",
        )

        chroms = scallion['chrom'].unique().tolist()
        positions = scallion['pos'].unique().tolist()
        refs = scallion['ref'].unique().tolist()
        alts = scallion['alt'].unique().tolist()

        filter_expr = (
            ds.field('chrom').isin(chroms) &
            ds.field('pos').isin(positions) &
            ds.field('ref').isin(refs) &
            ds.field('alt').isin(alts)
        )

        filtered_table = dataset.to_table(columns=VSMS_COLS, filter=filter_expr)
        scores = filtered_table.to_pandas()

        merged = scallion.merge(
            scores,
            on=['chrom', 'pos', 'ref', 'alt'],
            how='inner'
        )

        merged.to_csv(data_processed_path, sep='\t', index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)