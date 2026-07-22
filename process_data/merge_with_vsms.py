import hail as hl
import pyarrow.dataset as ds
import pandas as pd
import pickle
import argparse

from clinvar_enrichment import ( # Intended - Pass module via --pyfiles
    build_clinvar_ht,
    report_clinvar_priors,
    select_enriched_genes,
    DEFAULT_SC_GENES_ALL_PATH,
    DEFAULT_SC_GENES_MULTI_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_CLINVAR_HT_PATH,
    DEFAULT_HIGH_ENRICHMENT_OUTPUT,
    DEFAULT_LOW_ENRICHMENT_OUTPUT,
    MIN_MISSENSE_VARIANTS,
    N_GENES,
    Z,
)



VSMS_INNER_PATH = 'gs://grohlicek/genetics_gym_vsm_all_content/full_analysis_tables/gene_aggregated/variant_scores_all_outer_ensg_stats_eval.parquet'

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

CLINVAR_COLS = [
    'chrom', 'pos', 'ref', 'alt', 'rsid',
    'in_scallion_genes', 'gene_names_array', 'GENEINFO',
    'category', 'most_severe_consequence', 'variant_class',
    'CLNSIG_flat', 'CLNSIG_primary', 'CLNSIG_mapped', 'CLNSIG_is_simple',
    'CLNREVSTAT_flat', 'CLNREVSTAT_numeric',
]

DEFAULT_CLINVAR_VSM_OUTPUT_PATH = 'gs://aou_amc/scallion/benchmark/data/clinvar_w_vsm.tsv.gz'
DEFAULT_CLINVAR_PRIORS_OUTPUT_CSV = 'gs://aou_amc/data/utils/clinvar/clinvar_priors.csv'

DEFAULT_GENEBASS_VARS_PATH = 'gs://aou_amc/scallion/benchmark/data/genebass_allvars.tsv'
DEFAULT_GENEBASS_VSM_OUTPUT_PATH = 'gs://aou_amc/scallion/benchmark/data/genebass_w_vsm.tsv.gz'

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
        default=None,
        help="Legacy identifier used to locate the scallion results directory (e.g. 'run_2024_v1'). "
             "Required when --merge_type is 'vsm_scallion'; unused for 'clinvar_vsm'."
    )
    parser.add_argument(
        '--merge_type',
        type=str,
        required=True,
        choices=['vsm_scallion', 'clinvar_vsm', 'genebass_vsm'],
        help="Type of merge operation to perform: 'vsm_scallion', 'clinvar_vsm', or 'genebass_vsm'."
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help="If set, re-run and overwrite outputs even if they already exist."
    )
    parser.add_argument(
        '--spearman_vsm_scallion',
        action='store_true',
        help="If set (within merge_type='vsm_scallion'), run the VSM-vs-scallion "
             "statistical comparison and save the summary table and figure."
    )
    parser.add_argument(
        '--sc_genes_all_path',
        type=str,
        default=DEFAULT_SC_GENES_ALL_PATH,
        help="(merge_type='clinvar_vsm') Gene list used by build_clinvar_ht to flag in_scallion_genes."
    )
    parser.add_argument(
        '--clinvar_ht_path',
        type=str,
        default=DEFAULT_CLINVAR_HT_PATH,
        help="(merge_type='clinvar_vsm') Where to checkpoint/read the filtered ClinVar Hail Table."
    )
    parser.add_argument(
        '--clinvar_vsm_output_path',
        type=str,
        default=DEFAULT_CLINVAR_VSM_OUTPUT_PATH,
        help="(merge_type='clinvar_vsm') Where to write the ClinVar+VSM merged TSV."
    )
    parser.add_argument(
        '--generate_clinvar_priors',
        action='store_true',
        help="(merge_type='clinvar_vsm') Also run clinvar_enrichment's "
             "report step (pathogenic priors per functional category) on the ClinVar HT."
    )
    parser.add_argument(
        '--clinvar_priors_output_csv',
        type=str,
        default=DEFAULT_CLINVAR_PRIORS_OUTPUT_CSV,
        help="(--generate_clinvar_priors) Where to write the per-category priors CSV."
    )
    parser.add_argument(
        '--missense_enrichment',
        action='store_true',
        help="(merge_type='clinvar_vsm') Also run clinvar_enrichment's "
             "select step (high/low deleterious-missense-enrichment genes) on the ClinVar HT."
    )
    parser.add_argument(
        '--sc_genes_multi_path',
        type=str,
        default=DEFAULT_SC_GENES_MULTI_PATH,
        help="(--missense_enrichment) Scallion multi-gene list, used to label genes single vs. multi."
    )
    parser.add_argument(
        '--min_missense_variants',
        type=int,
        default=MIN_MISSENSE_VARIANTS,
        help="(--missense_enrichment) Floor on classified missense variants/gene before ranking."
    )
    parser.add_argument(
        '--n_genes',
        type=int,
        default=N_GENES,
        help="(--missense_enrichment) Number of genes to keep per scallion_label at each extreme."
    )
    parser.add_argument(
        '--z',
        type=float,
        default=Z,
        help="(--missense_enrichment) Z-score for the Wilson interval (1.96 ~= 95%%)."
    )
    parser.add_argument(
        '--high_enrichment_output_path',
        type=str,
        default=DEFAULT_HIGH_ENRICHMENT_OUTPUT,
        help="(--missense_enrichment) Where to export the high-enrichment gene table (TSV)."
    )
    parser.add_argument(
        '--low_enrichment_output_path',
        type=str,
        default=DEFAULT_LOW_ENRICHMENT_OUTPUT,
        help="(--missense_enrichment) Where to export the low-enrichment gene table (TSV)."
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
        if not scallion_prefix:
            raise ValueError("--scallion_prefix is required when --merge_type is 'vsm_scallion'")

        print(f"merge_type '{merge_type}' ")
        ensg_genesymbol_map     = "gs://aou_amc/scallion/utils/genesymbol_to_ensg.pkl"
        scallion_path           = f'gs://aou_amc/scallion/results_final/{scallion_prefix}/scallion_concatenated_qc.tsv'
        data_processed_tmp_path = f'gs://aou_amc/scallion/training/input_data/scallion_w_vsm_{scallion_prefix}_tmp.tsv.gz'
        data_processed_path     = f'gs://aou_amc/scallion/training/input_data/scallion_w_vsm_{scallion_prefix}.tsv.gz'
        vsms_inner              = VSMS_INNER_PATH

        print(f"scallion_path: {scallion_path}")
        print(f"data_processed_path: {data_processed_path}")

        need_merge = not hl.hadoop_exists(data_processed_path) or args.overwrite
        
        if need_merge:
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

            if not hl.hadoop_exists(data_processed_tmp_path) or args.overwrite:
                with hl.hadoop_open(ensg_genesymbol_map, 'rb') as f:
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

            dataset = ds.dataset(vsms_inner, format="parquet")

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

            merged = scallion.merge(scores, on=['chrom', 'pos', 'ref', 'alt'], how='inner')
            merged.to_csv(data_processed_path, sep='\t', index=False)

        elif args.spearman_vsm_scallion:
            print(f"{data_processed_path} already exists; loading it for the correlation analysis.")
            merged = pd.read_csv(data_processed_path, sep='\t')
            from correlation import compare_scores_by_scallion_group # Intended - Pass module via --pyfiles

            results_path = f'gs://aou_amc/scallion/results/{scallion_prefix}/vsms_vs_scallion/spearman_vsm_scallion.tsv'
            figure_path  = f'gs://aou_amc/scallion/results/{scallion_prefix}/vsms_vs_scallion/spearman_vsm_scallion.png'

            compare_scores_by_scallion_group(
                merged,
                prob_col='scallion_prob_mixture',
                results_path=results_path,
                figure_path=figure_path,
            )

    elif merge_type == 'clinvar_vsm':
        print(f"merge_type '{merge_type}' ")
        clinvar_ht_path      = args.clinvar_ht_path
        data_processed_path  = args.clinvar_vsm_output_path
        vsms_inner           = VSMS_INNER_PATH

        print(f"clinvar_ht_path: {clinvar_ht_path}")
        print(f"data_processed_path: {data_processed_path}")

        need_merge = not hl.hadoop_exists(data_processed_path) or args.overwrite
        need_ht = need_merge or args.generate_clinvar_priors or args.missense_enrichment

        ht = None
        if need_ht:
            if not hl.hadoop_exists(clinvar_ht_path) or args.overwrite:
                ht = build_clinvar_ht(
                    sc_genes_all_path=args.sc_genes_all_path,
                    output_path=clinvar_ht_path,
                    overwrite=args.overwrite,
                )
            else:
                ht = hl.read_table(clinvar_ht_path)

        if need_merge:
            variant_ht = ht.annotate(
                chrom=ht.locus.contig,
                pos=ht.locus.position,
                ref=ht.alleles[0],
                alt=ht.alleles[1],
            )
            clinvar = variant_ht.select(*CLINVAR_COLS).to_pandas()

            dataset = ds.dataset(vsms_inner, format="parquet")

            chroms = clinvar['chrom'].unique().tolist()
            positions = clinvar['pos'].unique().tolist()
            refs = clinvar['ref'].unique().tolist()
            alts = clinvar['alt'].unique().tolist()

            filter_expr = (
                ds.field('chrom').isin(chroms) &
                ds.field('pos').isin(positions) &
                ds.field('ref').isin(refs) &
                ds.field('alt').isin(alts)
            )

            filtered_table = dataset.to_table(columns=VSMS_COLS, filter=filter_expr)
            scores = filtered_table.to_pandas()

            merged = clinvar.merge(scores, on=['chrom', 'pos', 'ref', 'alt'], how='inner')
            merged.to_csv(data_processed_path, sep='\t', index=False)
        else:
            print(f"{data_processed_path} already exists; nothing to do.")

        if args.generate_clinvar_priors:
            print("Generating ClinVar pathogenic priors per functional category...")
            report_clinvar_priors(
                input_path=clinvar_ht_path,
                output_csv=args.clinvar_priors_output_csv,
            )

        if args.missense_enrichment:
            print("Selecting high/low deleterious-missense-enrichment genes...")
            high_enrichment, low_enrichment = select_enriched_genes(
                ht,
                sc_genes_all_path=args.sc_genes_all_path,
                sc_genes_multi_path=args.sc_genes_multi_path,
                min_missense_variants=args.min_missense_variants,
                n_genes=args.n_genes,
                z=args.z,
            )
            high_enrichment.export(args.high_enrichment_output_path)
            low_enrichment.export(args.low_enrichment_output_path)
            print(f"Wrote high-enrichment genes to {args.high_enrichment_output_path}")
            print(f"Wrote low-enrichment genes to {args.low_enrichment_output_path}")

    elif merge_type == 'genebass_vsm':
        print(f"merge_type '{merge_type}' ")
        genebass_vars_path  = DEFAULT_GENEBASS_VARS_PATH
        data_processed_path = DEFAULT_GENEBASS_VSM_OUTPUT_PATH
        vsms_inner          = VSMS_INNER_PATH

        print(f"genebass_vars_path: {genebass_vars_path}")
        print(f"data_processed_path: {data_processed_path}")

        need_merge = not hl.hadoop_exists(data_processed_path) or args.overwrite

        if need_merge:
            if not hl.hadoop_exists(genebass_vars_path) or args.overwrite:
                var_mt = hl.read_matrix_table('gs://ukbb-exome-public/500k/results/variant_results.mt')
                ht = var_mt.rows().select('gene')
                ht = ht.annotate(
                    chrom=ht.locus.contig,
                    pos=ht.locus.position,
                    ref=ht.alleles[0],
                    alt=ht.alleles[1],
                )
                ht.export(genebass_vars_path)

            genebass = pd.read_csv(genebass_vars_path, sep='\t')

            dataset = ds.dataset(vsms_inner, format="parquet")

            chroms = genebass['chrom'].unique().tolist()
            positions = genebass['pos'].unique().tolist()
            refs = genebass['ref'].unique().tolist()
            alts = genebass['alt'].unique().tolist()

            filter_expr = (
                ds.field('chrom').isin(chroms) &
                ds.field('pos').isin(positions) &
                ds.field('ref').isin(refs) &
                ds.field('alt').isin(alts)
            )

            filtered_table = dataset.to_table(columns=VSMS_COLS, filter=filter_expr)
            scores = filtered_table.to_pandas()

            merged = genebass.merge(scores, on=['chrom', 'pos', 'ref', 'alt'], how='inner')
            merged.to_csv(data_processed_path, sep='\t', index=False)
        else:
            print(f"{data_processed_path} already exists; nothing to do.")


if __name__ == '__main__':
    args = parse_args()
    main(args)