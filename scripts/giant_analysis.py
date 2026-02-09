__author__ = "ale"

import hail as hl
import pandas as pd
from gnomad.utils.vep import process_consequences

TMP_BUCKET = 'gs://aou_tmp/v8'
MY_BUCKET = 'gs://aou_amc/'


PLOF_CSQS = ["transcript_ablation", "splice_acceptor_variant",
             "splice_donor_variant", "stop_gained", "frameshift_variant"]

MISSENSE_CSQS = ["stop_lost", "start_lost", "transcript_amplification",
                 "inframe_insertion", "inframe_deletion", "missense_variant"]

SYNONYMOUS_CSQS = ["stop_retained_variant", "synonymous_variant"]

OTHER_CSQS = ["mature_miRNA_variant", "5_prime_UTR_variant",
              "3_prime_UTR_variant", "non_coding_transcript_exon_variant", "intron_variant",
              "NMD_transcript_variant", "non_coding_transcript_variant", "upstream_gene_variant",
              "downstream_gene_variant", "TFBS_ablation", "TFBS_amplification", "TF_binding_site_variant",
              "regulatory_region_ablation", "regulatory_region_amplification", "feature_elongation",
              "regulatory_region_variant", "feature_truncation", "intergenic_variant"]



def initialize_hail(qib_mode=False, log_file="/hail_operation_QoB.log", app_name=None, **kwargs):
    """
    Initialize Hail with optional batch-mode defaults and arbitrary Hail init kwargs.

    Parameters
    ----------
    batch_mode : bool
        Whether to use batch-style local cluster settings.
    log_file : str
        Log file path for Hail.
    app_name : str
        Optional Spark application name.
    **kwargs :
        Any additional arguments passed directly to hl.init().
    """
    
    init_params = {
        "tmp_dir": TMP_BUCKET,
        "gcs_requester_pays_configuration": 'aou-neale-gwas',
        "default_reference": "GRCh37",
        "log": log_file,
    }

    if app_name:
        init_params["app_name"] = app_name

    if qib_mode:
        init_params.update({
            "master": "local[32]",
            "worker_cores": 8,
            "worker_memory": "highmem"
        })

    init_params.update(kwargs)

    print(init_params)

    hl.init(**init_params)

def path_to_giant_to_ht(overwrite = True, threshold = 5e-8):
    
    def parse_maf(field):
        return hl.if_else(
            (field == "-") | hl.is_missing(field),
            hl.struct(
                var = hl.missing(hl.tstr),
                maf = hl.missing(hl.tfloat64)
            ),
            hl.struct(
                # take part before any comma first
                var = field.split(",")[0].split(":")[0],
                maf = hl.float64(field.split(",")[0].split(":")[1])
            )
        )
    
    if overwrite:
        # exome_data = pd.read_csv('https://giant-consortium.web.broadinstitute.org/images/d/d5/Height_EA_add_SV.txt.gz', sep = '\t')
        exome_data = pd.read_csv("https://portals.broadinstitute.org/collaboration/giant/images/5/59/Height_All_add_SV.txt.gz", sep = '\t')
        exome_data_filtered = exome_data[exome_data['Pvalue'] < threshold].reset_index(drop = True)
        exome_data_filtered['CHR'] = exome_data_filtered['CHR'].astype(str)
        giant_ht = hl.Table.from_pandas(exome_data_filtered)
        
        giant_ht = giant_ht.annotate(
            # EUR_parsed = parse_maf(giant_ht.EUR_MAF),
            # ExAC_parsed = parse_maf(giant_ht.ExAC_NFE_MAF)
            GMAF_parsed = parse_maf(giant_ht.GMAF),
            ExAC_MAF = parse_maf(giant_ht.ExAC_MAF)
        )
        
        path_out = 'gs://aou_amc/data/scallion/giant/tmp/giant_ht_exome_significant.ht'
        giant_ht.naive_coalesce(5).checkpoint(path_out, overwrite)
        
        return path_out
    
    else:
        return 'gs://aou_amc/data/scallion/giant/tmp/giant_ht_exome_significant.ht'

def get_munging_mapping(t: hl.Table,
                        *,
                        source_genome_build: str = 'GRCh37',
                        target_genome_build: str = 'GRCh37'
                       ) -> hl.Table:
    
    source_rg = hl.get_reference(source_genome_build)
    target_rg = hl.get_reference(target_genome_build)

    rg37 = hl.get_reference('GRCh37')
    rg37.add_sequence('gs://hail-common/references/human_g1k_v37.fasta.gz',
                      'gs://hail-common/references/human_g1k_v37.fasta.fai')

    rg38 = hl.get_reference('GRCh38')
    rg38.add_sequence('gs://hail-common/references/Homo_sapiens_assembly38.fasta.gz',
                      'gs://hail-common/references/Homo_sapiens_assembly38.fasta.fai')
    
    t = t.select(
        CHR = t.CHR,
        BP = hl.int32(t.POS),
        A1 = t.ALT,
        A2 = t.REF,)
    
    if source_genome_build != target_genome_build and source_genome_build == 'GRCh37' and target_genome_build == 'GRCh38':
        rg37.add_liftover('gs://hail-common/references/grch37_to_grch38.over.chain.gz', rg38)
        t = t.annotate(new_locus=hl.liftover(hl.locus(t.CHR, t.BP), rg38, include_strand=True))
        t = t.filter(hl.is_defined(t.new_locus))
        
    allele_conversion = hl.literal({'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'})
    t = t.annotate(
        a1_norm=hl.if_else(
            t.new_locus.is_negative_strand, 
            hl.str(t.A1).translate(allele_conversion),
            hl.str(t.A1)),
        a2_norm=hl.if_else(
            t.new_locus.is_negative_strand, 
            hl.str(t.A2).translate(allele_conversion), 
            hl.str(t.A2))
    )

    t = t.annotate(
        expected_ref=hl.get_sequence(
            t.new_locus.result.contig, 
            t.new_locus.result.position, 
            before=0, after=hl.len(t.a1_norm)-1, reference_genome=target_rg)
    )
    
    should_flip = (t.a1_norm == t.expected_ref)

    t = t.select(
        t.CHR, t.BP, t.A1, t.A2, t.a1_norm, t.a2_norm,
        remapped_locus=t.new_locus.result,
        remapped_alleles=hl.if_else(
            should_flip,
            t.a1_norm.split(',').extend(t.a2_norm.split(',')),
            t.a2_norm.split(',').extend(t.a1_norm.split(','))),
        should_flip=should_flip
    )

    t = t.persist()

    n_flipped = t.aggregate(hl.struct(total=hl.agg.count(), count=hl.agg.count_where(t.should_flip)))
    print(f'total {n_flipped.total} n_flipped {n_flipped.count}')
    
    if n_flipped.count > 100_000:
        n_init = t.count()
        t = t.filter(~hl.is_strand_ambiguous(t.a1_norm, t.a2_norm))
        n_not_strand_ambig = t.count()
        print(f'n_strand_ambig = {n_init - n_not_strand_ambig}')
        t = t.filter(hl.all(t.remapped_alleles.map(lambda x: x.length() == 1)))
        n_not_multiallelic = t.count()
        print(f'n_multiallelic = {n_not_strand_ambig - n_not_multiallelic}')
        
    return t

def munge_summary_stats(t: hl.Table, munging_mapping: hl.Table) -> hl.Table:
    t = t.select(
        RSID = t.SNPNAME,
        BETA = hl.float64(t.beta),
        A1_EFF_FRQ = hl.float64(t.GMAF_parsed.maf),
        SE = hl.float64(t.se),
        P = hl.float64(t.Pvalue),)
        # N =hl.float64(t.N))
    
    t = t.join(munging_mapping, how='inner')

    t = t.annotate(BETA=hl.if_else(t.should_flip, -1 * t.BETA, t.BETA),
                   A1_EFF_FRQ=hl.if_else(t.should_flip, 1 - t.A1_EFF_FRQ, t.A1_EFF_FRQ))
    
    t = t.filter(hl.is_defined(t.remapped_locus), keep=True)

    t = t.annotate(locus=t.remapped_locus, alleles=t.remapped_alleles)
    t = t.drop('remapped_locus', 'remapped_alleles', 'should_flip')

    t = t.key_by('locus', 'alleles')

    return t

def load_giant(pop = 'eur', genome_build = 'GRCh37', target_genome_build = 'GRCh38'):
    
    giant_path = path_to_giant_to_ht(overwrite=True, threshold=1e-6) 
    giant_ht = hl.read_table(giant_path).add_index(name='idx').key_by('idx')

    munging_mapping = get_munging_mapping(
        giant_ht,
        source_genome_build=genome_build,
        target_genome_build=target_genome_build
    )

    munging_mapping = munging_mapping.key_by('idx')
    
    munged_mt = munge_summary_stats(giant_ht, munging_mapping)
    
    return munged_mt

def load_vsms(all_vsm_path):
    all_vsm = hl.read_table(all_vsm_path)
    agg_vsm = all_vsm.group_by('locus', 'alleles').aggregate(
        gene_symbol = hl.agg.take(all_vsm.gene_symbol, 1)[0],

        proteinmpnn_llr_neg = hl.agg.mean(all_vsm.proteinmpnn_llr_neg),
        esm1b_neg           = hl.agg.mean(all_vsm.esm1b_neg),
        score_PAI3D        = hl.agg.mean(all_vsm.score_PAI3D),
        revel              = hl.agg.mean(all_vsm.revel),
        rasp_score         = hl.agg.mean(all_vsm.rasp_score),
        AM                 = hl.agg.mean(all_vsm.AM),
        MisFit_D           = hl.agg.mean(all_vsm.MisFit_D),
        MisFit_S           = hl.agg.mean(all_vsm.MisFit_S),
        polyphen_score     = hl.agg.mean(all_vsm.polyphen_score),
        cpt1_score         = hl.agg.mean(all_vsm.cpt1_score),
        popEVE_neg         = hl.agg.mean(all_vsm.popEVE_neg),
        EVE                = hl.agg.mean(all_vsm.EVE),
        ESM_1v_neg         = hl.agg.mean(all_vsm.ESM_1v_neg),
        mpc                = hl.agg.mean(all_vsm.mpc),
        cadd_score         = hl.agg.mean(all_vsm.cadd_score),
        gpn_msa_score      = hl.agg.mean(all_vsm.gpn_msa_score),
    )
    
    return agg_vsm


def annotation_case_builder(ht):
    
    case = hl.case(missing_false=True)
    
    case = (case
           .when(ht.lof == 'HC', 'pLoF')
           .when(ht.lof == 'LC', 'LC'))
    
    case = case.when(ht.most_severe_consequence == 'missense_variant', 'missense')
    
    case = case.when(hl.set(SYNONYMOUS_CSQS).contains(ht.most_severe_consequence), 'synonymous')
    
    case = case.when(hl.set(OTHER_CSQS).contains(ht.most_severe_consequence), 'non-coding')
    
    return case.or_missing()


def final_processing(ht):
    fields_drop = ['idx','a_index', 'was_split', 'old_locus', 
                   'old_alleles', 'context', 'coverage_mean', 
                   'coverage_10', 'coverage_20', 'methyl_mean',
                   'vep', 'vep_proc_id']

    ht_tc = ht.annotate(
        variant_class = ht.vep.variant_class
    ).drop(*fields_drop)

    ht_tc = ht_tc.explode(ht_tc.worst_csq_by_gene_canonical)

    ht_tc = ht_tc.filter( 
        (ht_tc.worst_csq_by_gene_canonical.canonical == 1) &
        (ht_tc.worst_csq_by_gene_canonical.gene_id.startswith('ENSG')))

    ht_tc = ht_tc.annotate(
        gene_id_gnomad = ht_tc.worst_csq_by_gene_canonical.gene_id,
        gene_symbol_gnomad= ht_tc.worst_csq_by_gene_canonical.gene_symbol,
        most_severe_consequence = ht_tc.worst_csq_by_gene_canonical.most_severe_consequence,
        transcript_id = ht_tc.worst_csq_by_gene_canonical.transcript_id,
        source = ht_tc.worst_csq_by_gene_canonical.source,
        lof = ht_tc.worst_csq_by_gene_canonical.lof,
    ).drop('worst_csq_by_gene_canonical')

    ht_tc = ht_tc.annotate(
        variant_id=ht_tc.locus.contig + ':' + 
        hl.str(ht_tc.locus.position) + ':' + 
        ht_tc.alleles[0] + ':' + 
        ht_tc.alleles[1],
        annotation=annotation_case_builder(ht_tc))

    ht_tc = ht_tc .key_by(
        'locus', 'alleles', 'gene_id_gnomad'
    )
    
    return ht_tc.naive_coalesce(5000)



def main():
    
    initialize_hail(
        # backend='batch',
        qib_mode=False,
        log_file="giant_vsms_gnomad_merge.log",
        app_name="Giant - VSMs - Gnomad merge",
    )

    # Load the data
    # giant_ht = process_giant(pop='eur')
    giant_ht = load_giant(pop = 'eur', genome_build = 'GRCh37', target_genome_build = 'GRCh38')

    all_vsm_path =  'gs://trisha-tmp/new_VSM_temp/vsm_all_tables/vsm_all_SNP_gene_lag.ht'
    all_vsm = load_vsms(all_vsm_path)

    giant_vsms = giant_ht.join(all_vsm, how='left')
    path_out = 'gs://aou_amc/data/scallion/giant/giant_vsms_tmp.ht'
    giant_vsms = giant_vsms.checkpoint(path_out, overwrite = True)
    print(f'Final giant vsm variants {giant_vsms.count()} ')

    # Gather gnomad information and final join
    from gnomad.utils.vep import process_consequences
    
    snp_vep_path = 'gs://gcp-public-data--gnomad/resources/context/grch38_context_vep_annotated.v105.ht'
    snp_vep_ht = hl.read_table(snp_vep_path)

    giant_vsm_gnomad = giant_vsms.join(snp_vep_ht, how='inner')
    process_vep_ht = process_consequences(giant_vsm_gnomad)
    giant_vsm_gnomad = giant_vsm_gnomad.annotate(
        worst_csq_by_gene_canonical=process_vep_ht[giant_vsm_gnomad.key].vep.worst_csq_by_gene_canonical
    )

    giant_vsm_gnomad_clean = final_processing(giant_vsm_gnomad)

    # giant_vsms_wgnomadvep = giant_vsm_gnomad.join(giant_vsms, how='inner')
    path_out = 'gs://aou_amc/data/scallion/giant/ALL_EXOME_significant_1e6_giant_vsms_gnomad_v2.ht'
    giant_vsm_gnomad_clean = giant_vsm_gnomad_clean.checkpoint(path_out, overwrite= True)
    print(f'Final variants {giant_vsm_gnomad_clean.count()}')
    print(f'Final distinct variants locus-alleles-geneid {giant_vsm_gnomad_clean.distinct().count()}')


if __name__ == "__main__":
    main()
