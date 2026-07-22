"""
Score a dataset with every finalized model under a models directory and
write a single TSV with one prediction column per model, added alongside
the original columns. Optionally also compute the pairwise Spearman
correlation between model predictions and plot it as a heatmap.

Run from the repo root:
    pixi run python -m model_predictions.predict_with_models \\
        --data-path model_predictions/data/scallion_benchmark_data_clinvar_w_vsm.tsv.gz \\
        --output-path model_predictions/predictions/scallion_benchmark_predictions.tsv.gz \\
        --models-dir model_training/models \\
        --correlation

--data-path/--output-path also accept gs:// URIs, e.g.:
    pixi run python -m model_predictions.predict_with_models \\
        --data-path gs://aou_amc/scallion/benchmark/data/genebass_w_vsm.tsv.gz \\
        --output-path gs://aou_amc/scallion/benchmark/data/genebass_w_vsm_w_predictions.tsv.gz

Run only the correlations too
pixi run python -m model_predictions.predict_with_models \
    --data-path gs://aou_amc/scallion/benchmark/data/genebass_w_vsm.tsv.gz \
    --output-path gs://aou_amc/scallion/benchmark/data/genebass_w_vsm_w_predictions.tsv.gz \
    --correlation \
    --correlation-output /Users/am3171/WorkDir/projects/aou_rvas/rvas_aou/model_predictions/reports/genebass/

Run to compute the percentile transformations
pixi run python -m model_predictions.predict_with_models \
    --data-path gs://aou_amc/scallion/benchmark/data/genebass_w_vsm.tsv.gz \
    --output-path gs://aou_amc/scallion/benchmark/data/genebass_w_vsm_w_predictions.tsv.gz \
    --pct-output-path gs://aou_amc/scallion/benchmark/data/genebass_w_vsm_w_predictions_w_pct.tsv.gz
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from model_training.transforms import TARGET_TRANSFORMS
from utils.io import ensure_parent_dir, path_exists
from utils.percentiles import run_percentiles
from utils.correlation import run_correlation

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "model_predictions/data/scallion_benchmark_data_clinvar_w_vsm.tsv.gz"
OUTPUT_PATH = REPO_ROOT / "model_predictions/predictions/scallion_benchmark_predictions_v2.tsv.gz"
PCT_OUTPUT_PATH = "gs://aou_amc/scallion/benchmark/data/genebass_w_vsm_w_predictions_w_pct.tsv.gz"

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


def predict_with_model(model_dir: Path, df: pd.DataFrame) -> pd.Series:
    metadata = json.load(open(model_dir / "model_metadata.json"))
    predictors = metadata["monotonic_features"]
    model = joblib.load(model_dir / "model.pkl")

    X = df[predictors]
    imputer_path = model_dir / "imputer.pkl"
    scaler_path = model_dir / "scaler.pkl"
    if imputer_path.exists():
        X = joblib.load(imputer_path).transform(X)
    if scaler_path.exists():
        X = joblib.load(scaler_path).transform(X)

    pred_model_scale = model.predict(X)
    inverse = TARGET_TRANSFORMS[metadata["target_transform"]][1]
    return pd.Series(inverse(pred_model_scale), index=df.index)


def resolve_model_dirs(models_dir: str) -> list:
    """--models-dir (auto-discovered) if given, else the curated MODEL_DIRS list."""
    if models_dir:
        return discover_model_dirs(Path(models_dir))
    return [REPO_ROOT / rel_dir for rel_dir in MODEL_DIRS]


def predict_all(data_path: str, model_dirs: list) -> tuple:
    """Score `data_path` with every model in `model_dirs`.

    Returns (df, pred_columns): `df` is the original data with one
    `pred_<model>` column added per model; `pred_columns` are the names of
    those added columns, in model order.
    """
    df = pd.read_csv(data_path, sep="\t", low_memory=False)
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
                        help="Tab-separated dataset to score (default: scallion benchmark data). "
                             "Accepts a local path or a gs:// URI.")
    parser.add_argument("--output-path", type=str, default=str(OUTPUT_PATH),
                        help="Where to write the input data with prediction columns added (tsv.gz). "
                             "Accepts a local path or a gs:// URI.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run predictions and overwrite --output-path even if it already exists. "
                             "If not set and --output-path already exists, predictions are skipped and "
                             "the existing output is reused for the correlation heatmap.")
    parser.add_argument("--pct-output-path", type=str, default=PCT_OUTPUT_PATH,
                        help="Where to write the gene-level percentile scores (tsv.gz). "
                             "Accepts a local path or a gs:// URI.")
    parser.add_argument("--overwrite-pct", action="store_true",
                        help="Re-run gene-level percentile computation and overwrite --pct-output-path "
                             "even if it already exists. If not set and --pct-output-path already exists, "
                             "percentile computation is skipped.")
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

    if path_exists(output_path) and not args.overwrite:
        print(f"{output_path} already exists; skipping prediction "
              f"(use --overwrite to re-run).")
        df = pd.read_csv(output_path, sep="\t", low_memory=False)
        pred_columns = [c for c in df.columns if c.startswith("pred_")]
    else:
        model_dirs = resolve_model_dirs(args.models_dir)
        df, pred_columns = predict_all(data_path, model_dirs)

        ensure_parent_dir(output_path)
        df.to_csv(output_path, sep="\t", index=False)
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
        run_percentiles(df, pct_output_path)


if __name__ == "__main__":
    main()
