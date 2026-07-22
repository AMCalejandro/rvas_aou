import pandas as pd

from utils.io import ensure_parent_dir

# Non-score identifier columns to keep (alongside the *_pct columns) in percentile output.
ID_COLUMNS = ["locus", "alleles", "gene", "chrom", "pos", "ref", "alt", "ensg"]


def add_percentiles_pd(df: pd.DataFrame, fields: list, gene_col: str = "gene") -> pd.DataFrame:
    """Add a gene-level percentile-rank column for each field in `fields`.

    Result is a float in [0.0, 1.0] — exact rank, not approximate.
    Missing values are mean-imputed before ranking: NaNs are filled with the
    gene's own mean for that field, falling back to the field's global mean
    for genes where every value is NaN. A field that is NaN for every row
    stays NaN throughout and propagates as NaN in the output.

    Parameters
    ----------
    df       : DataFrame with one row per variant
    fields   : column names to percentile-rank
    gene_col : column containing the gene grouping key

    Returns
    -------
    DataFrame with <field>_pct columns added (float64, [0.0, 1.0])
    """
    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise ValueError(f"add_percentiles_pd: fields not found in DataFrame: {missing}")

    for f in fields:
        gene_mean = df.groupby(gene_col)[f].transform("mean")
        imputed = df[f].fillna(gene_mean).fillna(df[f].mean())
        df[f"{f}_pct"] = (
            imputed.groupby(df[gene_col])
            .rank(method="average", na_option="keep", pct=True)
        )

    return df


def run_percentiles(df: pd.DataFrame, pct_output_path: str, gene_col: str = "gene"):
    """Gene-level percentile-rank every non-ID column (all scores plus pred_
    columns) and write the ID columns + the resulting *_pct columns."""
    score_fields = [c for c in df.columns if c not in ID_COLUMNS]
    df = add_percentiles_pd(df, score_fields, gene_col=gene_col)

    pct_columns = [f"{f}_pct" for f in score_fields]
    id_columns = [c for c in ID_COLUMNS if c in df.columns]
    out = df[id_columns + pct_columns]

    ensure_parent_dir(pct_output_path)
    out.to_csv(pct_output_path, sep="\t", index=False)
    print(f"Wrote {len(out)} rows x {len(out.columns)} columns -> {pct_output_path}")
