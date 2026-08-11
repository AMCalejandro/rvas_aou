"""
# Full run overwriting results
pixi run python -m model_predictions.predict_with_models \
    --data-path gs://aou_amc/scallion/benchmark/data/genebass_w_vsm.tsv.gz \
    --output-path gs://aou_amc/scallion/benchmark/data/genebass_w_vsm_w_predictions.tsv.gz \
    --pct-output-path gs://aou_amc/scallion/benchmark/data/genebass_w_vsm_w_predictions_w_pct.tsv.gz \
    --overwrite \
    --overwrite-pct \
    --models-dir /Users/am3171/WorkDir/projects/aou_rvas/rvas_aou/model_training/models \
    --correlation


pixi run python -m model_predictions.predict_with_models \
    --data-path gs://aou_amc/scallion/data/predictions/all_missense.parquet \
    --output-path gs://aou_amc/scallion/data/predictions/all_missense_w_predictions.parquet \
    --pct-output-path gs://aou_amc/scallion/data/predictions/all_missense_w_predictions_w_pct.parquet \
    --models-dir /Users/am3171/WorkDir/projects/aou_rvas/rvas_aou/model_training/models \
    --gene-col ensg
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from model_training.utils.transforms import TARGET_TRANSFORMS
from utils.io import ensure_parent_dir, path_exists
from utils.percentiles import run_percentiles
from utils.correlation import run_correlation

VSMS_COLS = [
    'chrom', 'pos', 'ref', 'alt', 'ensg',
    'AM', 'mcap', 'esm1b', 'gmvp', 'phylop', 'sift', 'cadd',
    'cpt', 'gpn_msa', 'ESM_1v', 'EVE', 'popEVE', 'PAI3D',
    'MisFit_D', 'MisFit_S', 'mpc', 'polyphen'
]

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "model_predictions/data/scallion_benchmark_data_clinvar_w_vsm.tsv.gz"
OUTPUT_PATH = REPO_ROOT / "model_predictions/predictions/scallion_benchmark_predictions_v2.tsv.gz"
PCT_OUTPUT_PATH = "gs://aou_amc/scallion/benchmark/data/genebass_w_vsm_w_predictions_w_pct.tsv"

MODEL_DIRS = [
    "model_training/models/scallion_prob_mixture_legacy_multi_keep_all_lightgbm_regressor",
    "model_training/models/scallion_prob_mixture_legacy_allgenes_keep_all_xgboost_regressor",

    "model_training/models/scallion_prob_mixture_deltascaled_multi_keep_all_xgboost_regressor",
    "model_training/models/scallion_prob_mixture_deltascaled_allgenes_keep_all_xgboost_regressor",

    "model_training/models/scallion_prob_mixture_clinvar_multi_keep_all_xgboost_regressor",
    "model_training/models/scallion_llr_clinvar_multi_keep_all_xgboost_regressor",
    "model_training/models/scallion_llr_keep_all",
    "model_training/models/scallion_llr_drop_conflicting",
    "model_training/models/scallion_llr_keep_all_elasticnet",
    "model_training/models/scallion_prob_mixture_clinvar_xgboost",
]


def discover_model_dirs(models_dir: Path) -> list:
    """Every immediate subdirectory of `models_dir` that holds a model.pkl."""
    model_dirs = sorted(p for p in models_dir.iterdir() if p.is_dir() and (p / "model.pkl").exists())
    if not model_dirs:
        raise ValueError(f"No models with a model.pkl found under {models_dir}")
    return model_dirs


def load_model_bundle(model_dir: Path) -> dict:
    """Load every on-disk artifact for a finalized model once, so it can be applied to many
    DataFrames without re-reading from disk each time."""
    metadata = json.load(open(model_dir / "model_metadata.json"))
    monotonic_features = metadata["monotonic_features"]
    # Pre-selection predictor set the imputer was actually fit on — for
    # linear models `monotonic_features` can be a dropped-down subset of
    # this (see finalize.py), so imputing must happen on the full set
    # first and only then narrow to `monotonic_features` for the scaler/model.
    full_predictors = metadata.get("predictors", monotonic_features)

    imputer_path = model_dir / "imputer.pkl"
    scaler_path = model_dir / "scaler.pkl"
    return {
        "model": joblib.load(model_dir / "model.pkl"),
        "monotonic_features": monotonic_features,
        "full_predictors": full_predictors,
        "imputer": joblib.load(imputer_path) if imputer_path.exists() else None,
        "scaler": joblib.load(scaler_path) if scaler_path.exists() else None,
        "inverse_transform": TARGET_TRANSFORMS[metadata["target_transform"]][1],
    }


def apply_model_bundle(bundle: dict, df: pd.DataFrame) -> pd.Series:
    monotonic_features = bundle["monotonic_features"]
    if bundle["imputer"] is not None:
        X = df[bundle["full_predictors"]]
        X = pd.DataFrame(
            bundle["imputer"].transform(X), columns=bundle["full_predictors"], index=df.index,
        )
        X = X[monotonic_features]
    else:
        X = df[monotonic_features]
    if bundle["scaler"] is not None:
        X = bundle["scaler"].transform(X)

    pred_model_scale = bundle["model"].predict(X)
    return pd.Series(bundle["inverse_transform"](pred_model_scale), index=df.index)


def predict_with_model(model_dir: Path, df: pd.DataFrame) -> pd.Series:
    return apply_model_bundle(load_model_bundle(model_dir), df)


def resolve_model_dirs(models_dir: str) -> list:
    """--models-dir (auto-discovered) if given, else the curated MODEL_DIRS list."""
    if models_dir:
        return discover_model_dirs(Path(models_dir))
    return [REPO_ROOT / rel_dir for rel_dir in MODEL_DIRS]


def read_table(path: str, columns: list = None) -> pd.DataFrame:
    """.parquet reads/writes parquet (with column projection, if given); anything else is
    tab-separated (optionally .gz)."""
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, sep="\t", low_memory=False)


def write_table(df: pd.DataFrame, path: str):
    ensure_parent_dir(path)
    if str(path).endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, sep="\t", index=False)


def predict_all(data_path: str, model_dirs: list, columns: list = None) -> tuple:
    """Score `data_path` with every model in `model_dirs`.

    Returns (df, pred_columns): `df` is the original data with one
    `pred_<model>` column added per model; `pred_columns` are the names of
    those added columns, in model order.
    """
    df = read_table(data_path, columns=columns)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {data_path}")

    pred_columns = []
    for model_dir in model_dirs:
        col_name = f"pred_{model_dir.name}"
        print(f"Predicting with '{model_dir.name}' -> column '{col_name}'")
        df[col_name] = predict_with_model(model_dir, df)
        pred_columns.append(col_name)
    return df, pred_columns


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH),
                        help="Dataset to score (default: scallion benchmark data). Accepts a "
                             "local path or a gs:// URI; .parquet reads parquet (with column "
                             "projection to VSMS_COLS), anything else is read as tsv[.gz].")
    parser.add_argument("--output-path", type=str, default=str(OUTPUT_PATH),
                        help="Where to write the input data with prediction columns added. "
                             "Accepts a local path or a gs:// URI; .parquet writes parquet, "
                             "otherwise tsv.gz.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run predictions and overwrite --output-path even if it already exists. "
                             "If not set and --output-path already exists, predictions are skipped and "
                             "the existing output is reused for the correlation heatmap.")
    parser.add_argument("--pct-output-path", type=str, default=PCT_OUTPUT_PATH,
                        help="Where to write the gene-level percentile scores. Accepts a local "
                             "path or a gs:// URI; .parquet writes parquet, otherwise tsv.")
    parser.add_argument("--overwrite-pct", action="store_true",
                        help="Re-run gene-level percentile computation and overwrite --pct-output-path "
                             "even if it already exists. If not set and --pct-output-path already exists, "
                             "percentile computation is skipped.")
    parser.add_argument("--gene-col", type=str, default="gene",
                        help="Column to group by for gene-level percentiles. Default: 'gene' "
                             "(genebass/clinvar merges). Raw VSM data (merge_type='vsm_all') has "
                             "no gene symbol column, only 'ensg' — pass --gene-col ensg for that.")
    parser.add_argument("--pct-columns", type=str, default=None,
                        help="Comma-separated list of columns to gene-level percentile-rank "
                             "(e.g. 'AM,mcap,pred_scallion_llr_keep_all'). Default: every non-ID "
                             "column (all raw scores + all pred_ columns).")
    parser.add_argument("--models-dir", type=str, default=None,
                        help="Directory with one subdirectory per finalized model — every subdirectory "
                             "with a model.pkl is used. Default: the curated MODEL_DIRS list.")
    parser.add_argument("--correlation", action="store_true",
                        help="Also compute the pairwise Spearman correlation between model predictions "
                             "and save it (CSV + heatmap PNG).")
    parser.add_argument("--correlation-output", type=str, default=None,
                        help="Where to save the correlation heatmap PNG (a sibling .csv is also written). "
                             "Default: model_predictions/reports/model_correlation_heatmap.png. "
                             "Implies --correlation.")
    return parser


def main():
    args = build_argparser().parse_args()

    data_path = args.data_path
    output_path = args.output_path
    pct_output_path = args.pct_output_path
    columns = VSMS_COLS if data_path.endswith(".parquet") else None

    if path_exists(output_path) and not args.overwrite:
        print(f"{output_path} already exists; skipping prediction "
              f"(use --overwrite to re-run).")
        df = read_table(output_path)
        pred_columns = [c for c in df.columns if c.startswith("pred_")]
    else:
        model_dirs = resolve_model_dirs(args.models_dir)
        df, pred_columns = predict_all(data_path, model_dirs, columns=columns)
        write_table(df, output_path)
        print(f"Wrote {len(df)} rows x {len(df.columns)} columns -> {output_path}")

    if args.correlation or args.correlation_output:
        correlation_output = Path(
            args.correlation_output
            or REPO_ROOT / "model_predictions/reports/model_correlation_heatmap.png"
        )
        run_correlation(df, pred_columns, correlation_output)

    if path_exists(pct_output_path) and not args.overwrite_pct:
        print(f"{pct_output_path} already exists; skipping percentile computation "
              f"(use --overwrite-pct to re-run).")
    else:
        pct_fields = [c.strip() for c in args.pct_columns.split(",")] if args.pct_columns else None
        run_percentiles(df, pct_output_path, gene_col=args.gene_col, fields=pct_fields)


if __name__ == "__main__":
    main()
