#!/usr/bin/env python3
"""
Quick comparison of SAIGE gene group files between an "original" GCS path and a
"new" GCS path produced by a re-run of export_gene_group_file() (amc_run_saige.py)
against newly-generated gene maps.

Group file line format (space-delimited, no header), one line per gene/tag:
    GENE_ID_GENE_SYMBOL var  chr:pos:ref:alt chr:pos:ref:alt ...
    GENE_ID_GENE_SYMBOL anno annotation annotation ...

(Only 'var' and 'anno' rows are present in the files being compared here --
there is no 'weight' row, so it is not checked.)

Usage:
    python3 compare_group_files.py \
        --orig gs://aou_analysis/v8/gene_results/bgen/EUR \
        --new  gs://aou_amc_analyses/results/gene_results/bgen/gnomad_context/unguided/EUR \
        --n-samples 20 --seed 0

Files are read straight from GCS via `gsutil cat` (no local copies are written).
"""
import argparse
import random
import subprocess
import sys
from collections import Counter, defaultdict


def gsutil_ls(path):
    path = path.rstrip('/') + '/'
    out = subprocess.run(['gsutil', 'ls', path], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"gsutil ls failed for {path}:\n{out.stderr}")
    return [l.strip() for l in out.stdout.splitlines() if l.strip()]


def gsutil_cat(path):
    out = subprocess.run(['gsutil', 'cat', path], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"gsutil cat failed for {path}:\n{out.stderr}")
    return out.stdout


def parse_group_file(text):
    """Return {gene: {tag: [values...]}} from group-file text."""
    genes = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(' ')
        gene, tag = parts[0], parts[1]
        values = [p for p in parts[2:] if p]
        genes.setdefault(gene, {})[tag] = values
    return genes


def compare_one(name, orig_text, new_text, show_examples=3):
    orig = parse_group_file(orig_text)
    new = parse_group_file(new_text)

    orig_genes, new_genes = set(orig), set(new)
    common = orig_genes & new_genes

    res = {
        'file': name,
        'n_genes_orig': len(orig_genes),
        'n_genes_new': len(new_genes),
        'n_genes_common': len(common),
        'genes_only_in_orig': sorted(orig_genes - new_genes),
        'genes_only_in_new': sorted(new_genes - orig_genes),
        'n_var_exact_match': 0,       # same variants, same order
        'n_var_set_match_diff_order': 0,  # same set of variants, different order
        'n_var_mismatch': 0,          # different variant sets entirely
        'n_var_mismatch_new_subset_of_orig': 0,  # new has no variants absent from orig
        'n_var_mismatch_new_has_novel': 0,       # new has >=1 variant absent from orig
        'n_anno_mismatch': 0,
        'anno_label_swaps': Counter(),  # (orig_label, new_label) -> count, for variants shared by both
        'mismatched_genes': [],
    }

    for gene in sorted(common):
        orig_vars = orig[gene].get('var', [])
        new_vars = new[gene].get('var', [])
        orig_set, new_set = set(orig_vars), set(new_vars)

        if orig_vars == new_vars:
            res['n_var_exact_match'] += 1
        elif orig_set == new_set:
            res['n_var_set_match_diff_order'] += 1
        else:
            res['n_var_mismatch'] += 1
            if new_set - orig_set:
                res['n_var_mismatch_new_has_novel'] += 1
            else:
                res['n_var_mismatch_new_subset_of_orig'] += 1
            if len(res['mismatched_genes']) < show_examples:
                res['mismatched_genes'].append({
                    'gene': gene,
                    'n_orig': len(orig_vars),
                    'n_new': len(new_vars),
                    'n_shared': len(orig_set & new_set),
                    'only_in_orig': sorted(orig_set - new_set)[:5],
                    'only_in_new': sorted(new_set - orig_set)[:5],
                })

        orig_anno = orig[gene].get('anno', [])
        new_anno = new[gene].get('anno', [])
        if orig_anno != new_anno:
            res['n_anno_mismatch'] += 1
            # per-variant label comparison, restricted to variants present in both
            # (anno[i] corresponds to var[i] within a gene)
            orig_label_by_var = dict(zip(orig_vars, orig_anno))
            new_label_by_var = dict(zip(new_vars, new_anno))
            for v in orig_set & new_set:
                ol, nl = orig_label_by_var.get(v), new_label_by_var.get(v)
                if ol != nl:
                    res['anno_label_swaps'][(ol, nl)] += 1

    return res


def print_report(results):
    print("=" * 90)
    print(f"Compared {len(results)} file pair(s)")
    print("=" * 90)

    totals = defaultdict(int)
    files_with_issues = []
    anno_label_swaps = Counter()

    for r in results:
        issues = (
            r['genes_only_in_orig'] or r['genes_only_in_new']
            or r['n_var_mismatch'] or r['n_var_set_match_diff_order']
            or r['n_anno_mismatch']
        )
        flag = "  " if not issues else "**"
        print(f"\n{flag} {r['file']}")
        print(f"    genes: orig={r['n_genes_orig']} new={r['n_genes_new']} "
              f"common={r['n_genes_common']} "
              f"only_orig={len(r['genes_only_in_orig'])} only_new={len(r['genes_only_in_new'])}")
        print(f"    var rows: exact_match={r['n_var_exact_match']} "
              f"same_set_diff_order={r['n_var_set_match_diff_order']} "
              f"mismatch={r['n_var_mismatch']}")
        print(f"    anno mismatches={r['n_anno_mismatch']}")

        if r['genes_only_in_orig']:
            print(f"    genes only in orig (up to 5): {r['genes_only_in_orig'][:5]}")
        if r['genes_only_in_new']:
            print(f"    genes only in new  (up to 5): {r['genes_only_in_new'][:5]}")
        for m in r['mismatched_genes']:
            print(f"    MISMATCH gene={m['gene']} n_orig={m['n_orig']} n_new={m['n_new']} "
                  f"n_shared={m['n_shared']} only_orig={m['only_in_orig']} only_new={m['only_in_new']}")

        if issues:
            files_with_issues.append(r['file'])
        anno_label_swaps.update(r['anno_label_swaps'])

        for k in ('n_genes_orig', 'n_genes_new', 'n_genes_common', 'n_var_exact_match',
                  'n_var_set_match_diff_order', 'n_var_mismatch',
                  'n_var_mismatch_new_subset_of_orig', 'n_var_mismatch_new_has_novel',
                  'n_anno_mismatch'):
            totals[k] += r[k]
        totals['n_genes_only_in_orig'] += len(r['genes_only_in_orig'])
        totals['n_genes_only_in_new'] += len(r['genes_only_in_new'])

    print("\n" + "=" * 90)
    print("SUMMARY ACROSS ALL SAMPLED FILES")
    print("=" * 90)
    for k, v in totals.items():
        print(f"  {k}: {v}")
    print(f"  files_with_any_issue: {len(files_with_issues)} / {len(results)}")
    if files_with_issues:
        print(f"  -> {files_with_issues}")

    if totals['n_var_mismatch']:
        print(f"\n  of {totals['n_var_mismatch']} genes with mismatched var sets:")
        print(f"    new is a strict subset of orig (new dropped variants, added none): "
              f"{totals['n_var_mismatch_new_subset_of_orig']}")
        print(f"    new has >=1 variant NOT in orig (novel/unexpected):               "
              f"{totals['n_var_mismatch_new_has_novel']}")

    if anno_label_swaps:
        print("\n  anno label swaps, for variants present in BOTH files (orig_label -> new_label): count")
        for (ol, nl), cnt in anno_label_swaps.most_common():
            print(f"    {ol!r} -> {nl!r}: {cnt}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--orig', default='gs://aou_analysis/v8/gene_results/bgen/EUR',
                     help='Original gene group file directory')
    ap.add_argument('--new', default='gs://aou_amc_analyses/results/gene_results/bgen/gnomad_context/unguided/EUR',
                     help='New gene group file directory to validate')
    ap.add_argument('--n-samples', type=int, default=10, help='Number of random shared files to compare')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--files', nargs='*', default=None,
                     help='Specific basenames (e.g. gene_chr10_000046892_22320308.gene.txt) to compare instead of random sampling')
    args = ap.parse_args()

    print(f"Listing {args.orig} ...")
    orig_files = {p.rsplit('/', 1)[-1]: p for p in gsutil_ls(args.orig) if p.endswith('.gene.txt')}
    print(f"Listing {args.new} ...")
    new_files = {p.rsplit('/', 1)[-1]: p for p in gsutil_ls(args.new) if p.endswith('.gene.txt')}

    common_names = sorted(set(orig_files) & set(new_files))
    only_orig = sorted(set(orig_files) - set(new_files))
    only_new = sorted(set(new_files) - set(orig_files))

    print(f"\n.gene.txt files found: orig={len(orig_files)} new={len(new_files)} "
          f"common={len(common_names)} only_in_orig={len(only_orig)} only_in_new={len(only_new)}")
    if only_orig:
        print(f"  only in orig (up to 10): {only_orig[:10]}")
    if only_new:
        print(f"  only in new  (up to 10): {only_new[:10]}")

    if not common_names:
        print("No shared .gene.txt filenames between the two paths -- nothing to compare.")
        sys.exit(1)

    if args.files:
        chosen = [f for f in args.files if f in common_names]
        missing = [f for f in args.files if f not in common_names]
        if missing:
            print(f"WARNING: requested files not found in both dirs, skipping: {missing}")
    else:
        rng = random.Random(args.seed)
        chosen = rng.sample(common_names, min(args.n_samples, len(common_names)))

    print(f"\nComparing {len(chosen)} file(s): {chosen}\n")

    results = []
    for name in chosen:
        orig_text = gsutil_cat(orig_files[name])
        new_text = gsutil_cat(new_files[name])
        results.append(compare_one(name, orig_text, new_text))

    print_report(results)


if __name__ == '__main__':
    main()
