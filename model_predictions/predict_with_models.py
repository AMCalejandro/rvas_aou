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

# Score all of VSMS_INNER_PATH (too large for one DataFrame) — first build the gene-hash-bucketed
# dataset with process_data/merge_with_vsms.py --merge_type vsm_all, then:
pixi run python -m model_predictions.predict_with_models \
    --data-path gs://aou_amc/scallion/data/predictions/all_missense.parquet/ \
    --output-path gs://aou_amc/scallion/data/predictions/all_missense_w_predictions.parquet \
    --pct-output-path gs://aou_amc/scallion/data/predictions/all_missense_w_predictions_w_pct.parquet \
    --models-dir /Users/am3171/WorkDir/projects/aou_rvas/rvas_aou/model_training/models \
    --bucketed --workers 32
# (buckets are scored in parallel into gs://.../all_missense_w_predictions.parquet.parts/ etc.,
# then consolidated into the single files above — a killed/preempted run can just be re-launched
# with the same command and will resume from whichever buckets are already staged.)
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from model_training.utils.transforms import TARGET_TRANSFORMS
from utils.io import ensure_parent_dir, path_exists
from utils.percentiles import ID_COLUMNS, add_percentiles_pd, run_percentiles
from utils.correlation import run_correlation

# Must match process_data/merge_with_vsms.py's VSMS_COLS — the column set the
# gene-hash-bucketed dataset (merge_type='vsm_all') was projected down to.
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


_worker_bundles = None  # populated once per worker process by _init_worker


def _init_worker(model_dirs: list):
    """Runs once per ProcessPoolExecutor worker (not once per bucket) — loading ~10 model
    bundles from disk is the expensive part, so every task handled by this worker reuses them."""
    global _worker_bundles
    _worker_bundles = [(model_dir.name, load_model_bundle(model_dir)) for model_dir in model_dirs]


def _process_bucket(frag_path: str, pred_path: str, pct_path: str) -> int:
    """Score one bucket file end to end and write both outputs — runs in a worker process.
    Buckets are fully independent (each is a complete gene group by construction of the
    gene-hash-bucketed dataset), so this needs no coordination with any other bucket."""
    df = pq.read_table(frag_path, columns=VSMS_COLS).to_pandas()
    if df.empty:
        return 0

    for name, bundle in _worker_bundles:
        df[f"pred_{name}"] = apply_model_bundle(bundle, df)

    ensure_parent_dir(pred_path)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), pred_path)

    score_fields = [c for c in df.columns if c not in ID_COLUMNS]
    df = add_percentiles_pd(df, score_fields, gene_col="ensg")
    pct_columns = [f"{f}_pct" for f in score_fields]
    id_columns = [c for c in ID_COLUMNS if c in df.columns]
    ensure_parent_dir(pct_path)
    pq.write_table(pa.Table.from_pandas(df[id_columns + pct_columns], preserve_index=False), pct_path)

    return len(df)


def _consolidate_parquet(parts_dir: str, final_path: str):
    """Merge a directory of part-*.parquet files into a single final parquet file. Streams one
    part at a time — writing each straight through to the output writer — so peak memory never
    exceeds a single bucket's size regardless of how many parts there are or how large the
    combined dataset is. Costs an extra full read+write pass over every row versus leaving the
    parts as-is; that's the deliberate tradeoff for ending up with one consumable file."""
    dataset = ds.dataset(parts_dir, format="parquet")
    fragments = list(dataset.get_fragments())
    print(f"Consolidating {len(fragments)} part files -> {final_path}")

    ensure_parent_dir(final_path)
    writer = None
    total_rows = 0
    try:
        for frag in fragments:
            table = frag.to_table()
            if writer is None:
                writer = pq.ParquetWriter(final_path, table.schema)
            writer.write_table(table)
            total_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    print(f"Wrote {total_rows} rows -> {final_path}")


def predict_bucketed(
    data_path: str, model_dirs: list, output_path: str, pct_output_path: str,
    overwrite: bool = False, overwrite_pct: bool = False, n_workers: int = None,
):
    """Score a gene-hash-bucketed parquet dataset (produced by merge_with_vsms.py's
    merge_type='vsm_all' — a Spark `df.repartition(n_buckets, 'ensg')` write, so it's a flat
    directory of part-*.parquet files, each one Spark partition) one bucket at a time, instead of
    loading everything into one DataFrame. Every variant for a given gene is guaranteed (by
    construction of the bucketed dataset) to live in a single file, so per-bucket gene-level
    percentile ranks are exact — identical to running `add_percentiles_pd` on the whole dataset
    at once.

    Buckets are fully independent, so they're scored in parallel across `n_workers` processes
    (default: os.cpu_count()), each bucket's result written to its own file under a
    `<output_path>.parts/`-style staging directory — this avoids serializing every bucket through
    one writer, and makes the run resumable (a bucket whose staged output files already exist is
    skipped, so a killed/preempted run can just be restarted). Once every bucket is scored, the
    staged parts are consolidated into the single `output_path`/`pct_output_path` files requested.
    """
    output_path = output_path.rstrip("/")
    pct_output_path = pct_output_path.rstrip("/")
    pred_parts_dir = f"{output_path}.parts"
    pct_parts_dir = f"{pct_output_path}.parts"

    if path_exists(output_path) and path_exists(pct_output_path) and not (overwrite or overwrite_pct):
        print(f"{output_path} and {pct_output_path} already exist; skipping "
              f"(use --overwrite/--overwrite-pct to re-run).")
        return

    dataset = ds.dataset(data_path, format="parquet")
    fragments = list(dataset.get_fragments())
    print(f"Found {len(fragments)} gene-hash buckets in {data_path}")

    tasks = []
    for i, frag in enumerate(fragments):
        pred_part = f"{pred_parts_dir}/part-{i:05d}.parquet"
        pct_part = f"{pct_parts_dir}/part-{i:05d}.parquet"
        already_done = (
            not overwrite and not overwrite_pct
            and path_exists(pred_part) and path_exists(pct_part)
        )
        if not already_done:
            tasks.append((frag.path, pred_part, pct_part))

    n_skipped = len(fragments) - len(tasks)
    if n_skipped:
        print(f"Skipping {n_skipped} already-scored buckets (resuming a prior run).")

    if tasks:
        n_workers = n_workers or os.cpu_count()
        print(f"Scoring {len(tasks)} buckets with {n_workers} worker processes...")

        total_rows = 0
        n_done = 0
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker, initargs=(model_dirs,)
        ) as pool:
            futures = {pool.submit(_process_bucket, *task): task for task in tasks}
            for future in as_completed(futures):
                _, pred_part, _ = futures[future]
                n_rows = future.result()
                n_done += 1
                total_rows += n_rows
                print(f"  [{n_done}/{len(tasks)}] {pred_part}: {n_rows} rows (total {total_rows})")
    else:
        print("All buckets already scored.")

    _consolidate_parquet(pred_parts_dir, output_path)
    _consolidate_parquet(pct_parts_dir, pct_output_path)


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
                        help="Where to write the gene-level percentile scores (tsv). "
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
    parser.add_argument("--bucketed", action="store_true",
                        help="--data-path is a gene-hash-bucketed parquet dataset (see "
                             "merge_with_vsms.py's merge_type='vsm_all') rather than a single tsv.gz. "
                             "Scores buckets in parallel (see --workers) instead of loading "
                             "everything into one DataFrame — for datasets too large for that (e.g. "
                             "all of VSMS_INNER_PATH). Each bucket's result is staged under "
                             "'<output-path>.parts/' (a bucket whose staged files already exist is "
                             "skipped, so a killed/preempted run can just be restarted), then all "
                             "parts are consolidated into the single parquet files at "
                             "--output-path/--pct-output-path. Not compatible with --correlation "
                             "(skipped at this scale).")
    parser.add_argument("--workers", type=int, default=None,
                        help="(--bucketed) Number of worker processes to score buckets in "
                             "parallel. Default: os.cpu_count().")
    return parser


def main():
    args = build_argparser().parse_args()

    data_path = args.data_path
    output_path = args.output_path
    pct_output_path = args.pct_output_path

    if args.bucketed:
        if args.correlation or args.correlation_output:
            raise ValueError(
                "--correlation is not supported together with --bucketed. Correlation needs the "
                "full dataset in memory, which --bucketed exists to avoid; run without --bucketed "
                "on a smaller sample if you need it."
            )
        model_dirs = resolve_model_dirs(args.models_dir)
        predict_bucketed(
            data_path, model_dirs, output_path, pct_output_path,
            overwrite=args.overwrite, overwrite_pct=args.overwrite_pct, n_workers=args.workers,
        )
        return

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
