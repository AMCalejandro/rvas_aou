#!/usr/bin/env python3
"""
Single model training script for Hail Batch execution.
Trains a specific classifier OR regressor with monotonicity constraints,
depending on the --framework argument:

  --framework binary      →  ClassifierBenchmark  (default)
  --framework regression  →  RegressorBenchmark

Saves the final trained model as a pickle file.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import time
import joblib


from .models_benchmark import (
    ClassifierBenchmark,
    RegressorBenchmark,
    MonotonicityEnforcer,
    MonotonicityType,
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import clone


VSM_PREDICTORS = [
    'AM', 'mcap', 'esm1b', 'gmvp', 'phylop', 'sift', 'cadd',
    'cpt', 'gpn_msa', 'ESM_1v', 'EVE', 'popEVE', 'PAI3D',
    'MisFit_D', 'MisFit_S', 'mpc', 'polyphen'
]

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            'Train a single classifier or regressor model '
            'with monotonicity constraints'
        )
    )

    # ── Positional ──────────────────────────────────────────────────────────
    parser.add_argument(
        'model_name', type=str,
        help=(
            'Name of the model to train '
            '(e.g. "XGBoost", "Ridge", "Logistic Regression")'
        )
    )
    parser.add_argument(
        'input_data', type=str,
        help='Path to input CSV/TSV file with features and target'
    )
    parser.add_argument(
        'output_folder', type=str,
        help='Path to output folder for results'
    )

    # ── Optional ─────────────────────────────────────────────────────────────
    parser.add_argument(
        '--target-column', type=str, default='target',
        help='Name of the target column (default: target)'
    )
    parser.add_argument(
        '--predictors-file', type=str, required=False,
        help='Path to a text file containing predictor names (one per line)'
    )
    parser.add_argument(
        '--framework', type=str, default='binary',
        choices=['binary', 'regression'],
        help=(
            'Training framework: '
            '"binary" for classification, '
            '"regression" for continuous target (default: binary)'
        )
    )

    # ── Binarisation options (binary framework only) ─────────────────────────
    threshold_group = parser.add_mutually_exclusive_group()
    threshold_group.add_argument(
        '--bin-threshold', type=float, default=None,
        help='Absolute threshold to binarize the target (e.g. 0.7)'
    )
    threshold_group.add_argument(
        '--top-percent', type=float, default=None,
        help='Use top N%% of values as class 1 (e.g. 5 for top 5%%)'
    )

    # ── CV / reproducibility ─────────────────────────────────────────────────
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--random-state', type=int, default=42,
        help='Random state for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--sep', type=str, default=',',
        help='Delimiter for input file (default: ,)'
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# def load_predictors_from_file(filepath: str) -> List[str]:
#     """Load predictor names from a text file (one per line)."""
#     if not os.path.exists(filepath):
#         raise FileNotFoundError(f"Predictor file not found: {filepath}")
#     with open(filepath) as f:
#         predictors = [
#             line.strip() for line in f
#             if line.strip() and not line.startswith('#')
#         ]
#     if not predictors:
#         raise ValueError("Predictor file is empty.")
#     print(f"Loaded {len(predictors)} predictors from {filepath}")
#     return predictors


def load_data(
    input_path: str,
    target_column: str,
    predictors: Optional[List[str]],
    framework: str,
    bin_thresh: Optional[float] = None,
    top_percent: Optional[float] = None,
    sep: str = ',',
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load training data and prepare features / target.

    For framework='binary'      → target is cast to int {0, 1}
                                   (binarised if continuous + threshold given)
    For framework='regression'  → target is kept as float; binarisation args
                                   are ignored.
    """
    print(f"Loading data from: {input_path}")

    if input_path.endswith('.tsv'):
        df = pd.read_csv(input_path, sep='\t')
    elif input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    else:
        df = pd.read_csv(input_path, sep=sep)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    if predictors is None:
        predictors = [c for c in df.columns if c != target_column]

    missing = [c for c in predictors if c not in df.columns]
    if missing:
        raise ValueError(f"Predictors not found in data: {missing}")

    X = df[predictors].copy()
    y = df[target_column].copy()

    unique_vals = set(y.dropna().unique())
    is_binary = unique_vals.issubset({0, 1})

    # ── Regression: keep target as float, no binarisation ───────────────────
    if framework == 'regression':
        if bin_thresh is not None or top_percent is not None:
            print(
                "Warning: --bin-threshold / --top-percent are ignored "
                "for framework='regression'."
            )
        y = y.astype(float)
        print(f"Regression target — mean={y.mean():.4f}, std={y.std():.4f}, "
              f"range=[{y.min():.4f}, {y.max():.4f}]")

    # ── Binary classification ────────────────────────────────────────────────
    elif framework == 'binary':
        if not is_binary:
            print("Binary framework with continuous target.")
            if bin_thresh is not None:
                print(f"Binarising using absolute threshold = {bin_thresh}")
                y = (y > bin_thresh).astype(int)
            elif top_percent is not None:
                if not (0 < top_percent < 100):
                    raise ValueError("top_percent must be between 0 and 100")
                cutoff = np.percentile(y.dropna(), 100 - top_percent)
                print(f"Binarising using top {top_percent}%  (cutoff={cutoff:.6f})")
                y = (y >= cutoff).astype(int)
            else:
                raise ValueError(
                    "Binary framework with continuous target requires "
                    "either --bin-threshold or --top-percent."
                )
        else:
            print("Target is already binary.")
            y = y.astype(int)
        print(f"Class distribution:\n{y.value_counts(normalize=True)}")

    else:
        raise ValueError(f"Unknown framework: {framework!r}")

    print(f"Using {len(X.columns)} predictors")
    return X, y


# ---------------------------------------------------------------------------
# Classifier trainer
# ---------------------------------------------------------------------------

class SingleModelTrainer(ClassifierBenchmark):
    """Single-model training wrapper around ClassifierBenchmark."""

    def run_single_model(self, X: pd.DataFrame, y: pd.Series,
                         model_name: str) -> Dict:
        feature_names = list(X.columns)
        models = self.get_models(feature_names)

        if model_name not in models:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {list(models.keys())}"
            )

        model = models[model_name]

        print("=" * 80)
        print(f"TRAINING CLASSIFIER: {model_name}")
        print("=" * 80)
        print(f"  Samples:    {len(X)}")
        print(f"  Features:   {len(X.columns)}")
        print(f"  Positive:   {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
        print(f"  Negative:   {(~y.astype(bool)).sum()} "
              f"({(~y.astype(bool)).sum()/len(y)*100:.2f}%)")
        print(f"  CV folds:   {self.n_folds}")

        start = time.time()
        results = self.evaluate_model_cv(X, y, model, model_name)
        results['training_time'] = time.time() - start

        print(f"\n{'='*80}\nFINAL RESULTS SUMMARY\n{'='*80}")
        print(f"  Average Precision : {results['avg_precision_mean']:.4f} ± {results['avg_precision_std']:.4f}")
        print(f"  ROC-AUC           : {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
        print(f"  F1 Score          : {results['f1_score_mean']:.4f} ± {results['f1_score_std']:.4f}")
        print(f"  Precision         : {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
        print(f"  Recall            : {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
        print(f"  Monotonic features: {results.get('n_monotonic_features', len(X.columns))}/{len(X.columns)}")
        print(f"  Training time     : {results['training_time']:.2f}s")

        return {
            'framework': 'binary',
            'model_performance': results,
            'dataset_info': {
                'n_samples':      len(X),
                'n_features':     len(X.columns),
                'feature_names':  list(X.columns),
                'n_positive':     int(y.sum()),
                'n_negative':     int((~y.astype(bool)).sum()),
                'positive_rate':  float(y.sum() / len(y)),
            },
            'monotonicity_info': {
                'uses_native_monotonicity': results.get('uses_native_monotonicity', False),
                'n_monotonic_features':     results.get('n_monotonic_features', len(X.columns)),
                'monotonic_features':       results.get('monotonic_features', list(X.columns)),
                'monotonic_feature_frequency': results.get('monotonic_feature_frequency', {}),
                'n_features_mean':          results.get('n_features_mean', len(X.columns)),
                'n_features_std':           results.get('n_features_std', 0.0),
            },
        }

    def train_final_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        monotonic_features: List[str],
    ) -> Tuple[object, Optional[StandardScaler]]:
        """Train final classifier on all data using CV-selected features."""
        print(f"\n{'='*80}\nTRAINING FINAL CLASSIFIER ON ALL DATA\n{'='*80}")

        feature_names = list(X.columns)
        models = self.get_models(feature_names)
        final_model = clone(models[model_name])

        X_filtered = X[monotonic_features].copy()
        print(f"Using {len(monotonic_features)} monotonic features: {monotonic_features}")

        # Class-imbalance weight for tree models
        n_pos = int(y.sum())
        n_neg = int((~y.astype(bool)).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        if model_name in {'XGBoost', 'XGBoost (Deep)'}:
            final_model.set_params(scale_pos_weight=scale_pos_weight)
        if model_name in {'LightGBM', 'LightGBM (Deep)'}:
            final_model.set_params(scale_pos_weight=scale_pos_weight)

        needs_scaling = model_name not in {
            'XGBoost', 'LightGBM', 'Random Forest',
            'XGBoost (Deep)', 'LightGBM (Deep)',
        }

        scaler = None
        if needs_scaling:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_filtered)
            if model_name == 'Linear SVM':
                final_model.fit(X_scaled, y)
                calibrator = CalibratedClassifierCV(
                    estimator=final_model, method='sigmoid', cv=3
                )
                calibrator.fit(X_scaled, y)
                final_model = calibrator
            else:
                final_model.fit(X_scaled, y)
        else:
            final_model.fit(X_filtered.values, y)

        print("Final classifier training completed.")
        return final_model, scaler, monotonic_features 


# ---------------------------------------------------------------------------
# Regressor trainer
# ---------------------------------------------------------------------------

class SingleRegressorTrainer(RegressorBenchmark):
    """Single-model training wrapper around RegressorBenchmark."""

    def run_single_model(self, X: pd.DataFrame, y: pd.Series,
                         model_name: str) -> Dict:
        feature_names = list(X.columns)
        models = self.get_models(feature_names)

        if model_name not in models:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {list(models.keys())}"
            )

        model = models[model_name]

        print("=" * 80)
        print(f"TRAINING REGRESSOR: {model_name}")
        print("=" * 80)
        print(f"  Samples:  {len(X)}")
        print(f"  Features: {len(X.columns)}")
        print(f"  Target — mean={y.mean():.4f}, std={y.std():.4f}, "
              f"range=[{y.min():.4f}, {y.max():.4f}]")
        print(f"  CV folds: {self.n_folds}")

        start = time.time()
        results = self.evaluate_model_cv(X, y, model, model_name)
        results['training_time'] = time.time() - start

        print(f"\n{'='*80}\nFINAL RESULTS SUMMARY\n{'='*80}")
        print(f"  RMSE              : {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
        print(f"  MAE               : {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
        print(f"  R²                : {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
        print(f"  Spearman ρ        : {results['spearman_mean']:.4f} ± {results['spearman_std']:.4f}")
        print(f"  Monotonic features: {results.get('n_monotonic_features', len(X.columns))}/{len(X.columns)}")
        print(f"  Training time     : {results['training_time']:.2f}s")

        return {
            'framework': 'regression',
            'model_performance': results,
            'dataset_info': {
                'n_samples':    len(X),
                'n_features':   len(X.columns),
                'feature_names': list(X.columns),
                'target_mean':  float(y.mean()),
                'target_std':   float(y.std()),
                'target_min':   float(y.min()),
                'target_max':   float(y.max()),
            },
            'monotonicity_info': {
                'uses_native_monotonicity': results.get('uses_native_monotonicity', False),
                'n_monotonic_features':     results.get('n_monotonic_features', len(X.columns)),
                'monotonic_features':       results.get('monotonic_features', list(X.columns)),
                'monotonic_feature_frequency': results.get('monotonic_feature_frequency', {}),
                'n_features_mean':          results.get('n_features_mean', len(X.columns)),
                'n_features_std':           results.get('n_features_std', 0.0),
            },
        }

    # def train_final_model(
    #     self,
    #     X: pd.DataFrame,
    #     y: pd.Series,
    #     model_name: str,
    #     monotonic_features: List[str],
    # ) -> Tuple[object, Optional[StandardScaler]]:
    #     """Train final regressor on all data using CV-selected features."""
    #     print(f"\n{'='*80}\nTRAINING FINAL REGRESSOR ON ALL DATA\n{'='*80}")

    #     feature_names = list(X.columns)
    #     models = self.get_models(feature_names)
    #     final_model = clone(models[model_name])

    #     X_filtered = X[monotonic_features].copy()
    #     print(f"Using {len(monotonic_features)} monotonic features: {monotonic_features}")

    #     needs_scaling = self._needs_scaling(model_name)

    #     scaler = None
    #     if needs_scaling:
    #         scaler = StandardScaler()
    #         X_scaled = scaler.fit_transform(X_filtered)
    #         final_model.fit(X_scaled, y)
    #     else:
    #         final_model.fit(X_filtered.values, y)

    #     print("Final regressor training completed.")
    #     return final_model, scaler
    def train_final_model(self, X: pd.DataFrame, y: pd.Series,
                      model_name: str,
                      monotonic_features: List[str]) -> Tuple[object, Optional[StandardScaler]]:

        feature_names = list(X.columns)
        models = self.get_models(feature_names)
        final_model = clone(models[model_name])

        X_filtered = X[monotonic_features].copy()

        needs_scaling = model_name not in ['XGBoost', 'LightGBM', 'Random Forest',
                                            'XGBoost (Deep)', 'LightGBM (Deep)']

        enforcer = MonotonicityEnforcer()
        scaler = None

        if self.uses_native_monotonicity(model_name):
            if needs_scaling:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_filtered)
                final_model.fit(X_scaled, y)
            else:
                final_model.fit(X_filtered, y)

        elif enforcer.supports_coefficient_check(model_name):
            # Re-run iterative selection on the full dataset — CV-selected features
            # are NOT guaranteed to have positive coefficients when retrained on all data.
            final_features, final_model, scaler = \
                MonotonicityEnforcer.iterative_monotonic_selection(
                    X_filtered,
                    y,
                    final_model,
                    list(X_filtered.columns),
                    needs_scaling=needs_scaling,
                    threshold=0.0
                )

            if set(final_features) != set(monotonic_features):
                dropped = set(monotonic_features) - set(final_features)
                print(f"\n  WARNING: iterative_monotonic_selection on full data "
                    f"dropped additional features: {dropped}")

            X_filtered = X_filtered[final_features]

            if model_name == "Linear SVM":
                calibrator = CalibratedClassifierCV(
                    estimator=final_model,
                    method="sigmoid",
                    cv=3
                )
                X_for_calib = scaler.transform(X_filtered) if scaler else X_filtered.values
                calibrator.fit(X_for_calib, y)
                final_model = calibrator

            # Update monotonic_features to reflect what was actually used
            monotonic_features = final_features

        else:
            # Marginal-only models (Random Forest, GaussianNB):
            # Spearman filter on full data as a consistency check
            marginal_scores = enforcer.check_monotonicity(X_filtered, y)
            final_features = enforcer.get_monotonic_features(marginal_scores, threshold=0.0)

            if not final_features:
                final_features = monotonic_features  # fallback

            if set(final_features) != set(monotonic_features):
                dropped = set(monotonic_features) - set(final_features)
                print(f"\n  WARNING: Marginal re-check on full data "
                    f"dropped additional features: {dropped}")

            X_filtered = X_filtered[final_features]
            monotonic_features = final_features

            if needs_scaling:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_filtered)
                final_model.fit(X_scaled, y)
            else:
                final_model.fit(X_filtered, y)

        print(f"Final model training completed")
        print(f"Final monotonic features ({len(monotonic_features)}): {monotonic_features}")

        return final_model, scaler, monotonic_features  # ← also return updated features

# ---------------------------------------------------------------------------
# Saving utilities  (framework-aware)
# ---------------------------------------------------------------------------
import numpy as np
from sklearn.metrics import precision_recall_curve


def _interpolate_mean_pr_curve(
    pr_curve_folds: List[Dict],
    n_points: int = 200,
) -> Dict:
    """
    Interpolate each fold's PR curve onto a shared recall grid and
    compute the mean ± std precision across folds.

    Recall is naturally decreasing from precision_recall_curve, so we
    flip, interpolate on an ascending grid, then report the result in
    descending recall order (conventional PR-curve orientation).
    """
    recall_grid = np.linspace(0, 1, n_points)
    interp_precisions = []

    for fold_data in pr_curve_folds:
        rec  = np.array(fold_data['recall'])
        prec = np.array(fold_data['precision'])
        # precision_recall_curve returns descending recall → flip to ascending
        rec_asc  = rec[::-1]
        prec_asc = prec[::-1]
        # np.interp needs strictly increasing x; deduplicate
        _, idx = np.unique(rec_asc, return_index=True)
        interp_p = np.interp(recall_grid, rec_asc[idx], prec_asc[idx])
        interp_precisions.append(interp_p)

    interp_precisions = np.array(interp_precisions)   # shape (n_folds, n_points)

    return {
        'recall_grid':     recall_grid.tolist(),
        'precision_mean':  interp_precisions.mean(axis=0).tolist(),
        'precision_std':   interp_precisions.std(axis=0).tolist(),
        'precision_upper': (interp_precisions.mean(axis=0) + interp_precisions.std(axis=0)).tolist(),
        'precision_lower': np.clip(
            interp_precisions.mean(axis=0) - interp_precisions.std(axis=0), 0, 1
        ).tolist(),
    }


def _summarise_scatter_folds(scatter_folds: List[Dict]) -> Dict:
    """
    Collect all fold actual/predicted pairs and compute per-point
    residuals for an actual-vs-predicted plot.
    """
    all_true, all_pred, all_fold_ids = [], [], []
    for fd in scatter_folds:
        all_true.extend(fd['y_true'])
        all_pred.extend(fd['y_pred'])
        all_fold_ids.extend([fd['fold']] * len(fd['y_true']))

    all_true  = np.array(all_true)
    all_pred  = np.array(all_pred)
    residuals = all_true - all_pred

    return {
        'y_true':     all_true.tolist(),
        'y_pred':     all_pred.tolist(),
        'residuals':  residuals.tolist(),
        'fold_ids':   all_fold_ids,
        # Diagonal reference line boundaries
        'axis_min':   float(min(all_true.min(), all_pred.min())),
        'axis_max':   float(max(all_true.max(), all_pred.max())),
    }

def save_results(results: Dict, output_folder: str, model_name: str):
    """Save results to output folder. Handles both binary and regression."""

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    framework   = results.get('framework', 'binary')
    perf        = results['model_performance']
    mono_info   = results['monotonicity_info']
    dataset_info = results['dataset_info']

    # ── JSON ────────────────────────────────────────────────────────────────
    def _convert(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, dict):        return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):        return [_convert(i) for i in obj]
        return obj

    json_path = output_path / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(_convert(results), f, indent=2)
    print(f"Results saved to: {json_path}")

    # ── Human-readable summary ───────────────────────────────────────────────
    summary_path = output_path / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"MODEL TRAINING RESULTS: {model_name}  [{framework.upper()}]\n")
        f.write("=" * 80 + "\n\n")

        # Dataset block
        f.write("DATASET INFORMATION\n" + "-" * 80 + "\n")
        f.write(f"Total samples:  {dataset_info['n_samples']}\n")
        f.write(f"Total features: {dataset_info['n_features']}\n")
        if framework == 'binary':
            f.write(f"Positive class: {dataset_info['n_positive']} "
                    f"({dataset_info['positive_rate']*100:.2f}%)\n")
            f.write(f"Negative class: {dataset_info['n_negative']} "
                    f"({(1-dataset_info['positive_rate'])*100:.2f}%)\n\n")
        else:
            f.write(f"Target mean:  {dataset_info['target_mean']:.4f}\n")
            f.write(f"Target std:   {dataset_info['target_std']:.4f}\n")
            f.write(f"Target range: [{dataset_info['target_min']:.4f}, "
                    f"{dataset_info['target_max']:.4f}]\n\n")

        # Monotonicity block
        f.write("MONOTONICITY INFORMATION\n" + "-" * 80 + "\n")
        f.write(f"Uses native monotonicity:   {mono_info['uses_native_monotonicity']}\n")
        f.write(f"Stable monotonic features:  "
                f"{mono_info['n_monotonic_features']}/{dataset_info['n_features']}\n")
        f.write(f"Average features per fold:  "
                f"{mono_info['n_features_mean']:.1f} ± {mono_info['n_features_std']:.1f}\n")

        if mono_info['monotonic_features']:
            f.write("\nMonotonic features (stable in >=50% of folds):\n")
            for feat in sorted(mono_info['monotonic_features']):
                freq = mono_info['monotonic_feature_frequency'].get(feat, 1.0)
                f.write(f"  - {feat}  (selected in {freq*100:.0f}% of folds)\n")

        excluded = set(dataset_info['feature_names']) - set(mono_info['monotonic_features'])
        if excluded:
            f.write("\nExcluded features (non-monotonic or unstable):\n")
            for feat in sorted(excluded):
                freq = mono_info['monotonic_feature_frequency'].get(feat, 0.0)
                f.write(f"  - {feat}  (selected in {freq*100:.0f}% of folds)\n")
        f.write("\n")

        # Performance block
        f.write("PERFORMANCE METRICS\n" + "-" * 80 + "\n")
        if framework == 'binary':
            f.write(f"Average Precision : {perf['avg_precision_mean']:.4f} ± {perf['avg_precision_std']:.4f}\n")
            f.write(f"ROC-AUC           : {perf['roc_auc_mean']:.4f} ± {perf['roc_auc_std']:.4f}\n")
            f.write(f"F1 Score          : {perf['f1_score_mean']:.4f} ± {perf['f1_score_std']:.4f}\n")
            f.write(f"Precision         : {perf['precision_mean']:.4f} ± {perf['precision_std']:.4f}\n")
            f.write(f"Recall            : {perf['recall_mean']:.4f} ± {perf['recall_std']:.4f}\n")
            f.write(f"\nConfusion Matrix (averaged):\n")
            f.write(f"  TN: {perf['tn_mean']:.1f}  FP: {perf['fp_mean']:.1f}\n")
            f.write(f"  FN: {perf['fn_mean']:.1f}  TP: {perf['tp_mean']:.1f}\n")
        else:
            f.write(f"RMSE       : {perf['rmse_mean']:.4f} ± {perf['rmse_std']:.4f}\n")
            f.write(f"MAE        : {perf['mae_mean']:.4f} ± {perf['mae_std']:.4f}\n")
            f.write(f"R²         : {perf['r2_mean']:.4f} ± {perf['r2_std']:.4f}\n")
            f.write(f"Spearman ρ : {perf['spearman_mean']:.4f} ± {perf['spearman_std']:.4f}\n")

        f.write(f"\nTraining time: {perf['training_time']:.2f}s\n")

    print(f"Summary saved to: {summary_path}")

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = output_path / 'metrics.csv'
    row = {
        'model':                    model_name,
        'framework':                framework,
        'training_time':            perf['training_time'],
        'uses_native_monotonicity': mono_info['uses_native_monotonicity'],
        'n_monotonic_features':     mono_info['n_monotonic_features'],
        'n_features_mean':          mono_info['n_features_mean'],
        'n_features_std':           mono_info['n_features_std'],
    }
    if framework == 'binary':
        row.update({
            'avg_precision_mean': perf['avg_precision_mean'],
            'avg_precision_std':  perf['avg_precision_std'],
            'roc_auc_mean':       perf['roc_auc_mean'],
            'roc_auc_std':        perf['roc_auc_std'],
            'f1_score_mean':      perf['f1_score_mean'],
            'f1_score_std':       perf['f1_score_std'],
            'precision_mean':     perf['precision_mean'],
            'recall_mean':        perf['recall_mean'],
        })
    else:
        row.update({
            'rmse_mean':     perf['rmse_mean'],
            'rmse_std':      perf['rmse_std'],
            'mae_mean':      perf['mae_mean'],
            'mae_std':       perf['mae_std'],
            'r2_mean':       perf['r2_mean'],
            'r2_std':        perf['r2_std'],
            'spearman_mean': perf['spearman_mean'],
            'spearman_std':  perf['spearman_std'],
        })

    # ── Curve data for downstream plotting ──────────────────────────────────
    perf = results['model_performance']   # already defined above in the function

    if framework == 'binary' and 'pr_curve_folds' in perf:
        # Per-fold raw curves
        pr_folds_path = output_path / 'pr_curve_folds.json'
        with open(pr_folds_path, 'w') as f:
            json.dump(_convert(perf['pr_curve_folds']), f)
        print(f"PR curve folds saved to: {pr_folds_path}")

        # Mean ± std interpolated curve
        mean_pr = _interpolate_mean_pr_curve(perf['pr_curve_folds'])
        mean_pr_path = output_path / 'pr_curve_mean.json'
        with open(mean_pr_path, 'w') as f:
            json.dump(_convert(mean_pr), f, indent=2)
        print(f"Mean PR curve saved to:  {mean_pr_path}")

    if framework == 'regression' and 'scatter_folds' in perf:
        # Per-fold raw arrays
        scatter_folds_path = output_path / 'scatter_folds.json'
        with open(scatter_folds_path, 'w') as f:
            json.dump(_convert(perf['scatter_folds']), f)
        print(f"Scatter folds saved to:  {scatter_folds_path}")

        # Aggregated actual-vs-predicted + residuals
        scatter_summary = _summarise_scatter_folds(perf['scatter_folds'])
        scatter_path = output_path / 'actual_vs_predicted.json'
        with open(scatter_path, 'w') as f:
            json.dump(_convert(scatter_summary), f, indent=2)
        print(f"Actual vs predicted saved to: {scatter_path}")

    pd.DataFrame([row]).to_csv(csv_path, index=False)
    print(f"Metrics CSV saved to: {csv_path}")



def save_model(
    model: object,
    scaler: Optional[StandardScaler],
    monotonic_features: List[str],
    model_name: str,
    output_folder: str,
    framework: str = 'binary',
):
    """Save trained model, scaler, feature list and metadata."""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = output_path / 'model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # Scaler (if any)
    if scaler is not None:
        scaler_path = output_path / 'scaler.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")

    # Metadata JSON
    metadata = {
        'model_name':         model_name,
        'framework':          framework,
        'monotonic_features': monotonic_features,
        'n_features':         len(monotonic_features),
        'uses_scaler':        scaler is not None,
    }
    metadata_path = output_path / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_path}")

    # Plain-text feature list
    features_path = output_path / 'features.txt'
    with open(features_path, 'w') as f:
        f.write("# Monotonic features used in the final model\n")
        f.write("# One feature per line\n\n")
        for feat in monotonic_features:
            f.write(f"{feat}\n")
    print(f"Feature list saved to: {features_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()

    print("=" * 80)
    print("SINGLE MODEL TRAINING SCRIPT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model:         {args.model_name}")
    print(f"  Framework:     {args.framework}")
    print(f"  Input data:    {args.input_data}")
    print(f"  Output folder: {args.output_folder}")
    print(f"  Target column: {args.target_column}")
    if args.framework == 'binary':
        print(f"  Bin threshold: {args.bin_threshold}")
        print(f"  Top percent:   {args.top_percent}")
    print(f"  CV folds:      {args.n_folds}")
    print(f"  Random state:  {args.random_state}")
    print()

    try:
        # ── Load predictors ──────────────────────────────────────────────────
        predictors = (
            [p.strip() for p in args.predictors.split(',') if p.strip()]
            if args.predictors
            else VSM_PREDICTORS
        )

        # ── Load data ────────────────────────────────────────────────────────
        X, y = load_data(
            args.input_data, args.target_column, predictors,
            args.framework,
            bin_thresh=args.bin_threshold,
            top_percent=args.top_percent,
            sep=args.sep,
        )

        # ── Branch: classifier vs regressor ──────────────────────────────────
        if args.framework == 'binary':
            trainer = SingleModelTrainer(
                n_folds=args.n_folds, random_state=args.random_state
            )
        else:  # regression
            trainer = SingleRegressorTrainer(
                n_folds=args.n_folds, random_state=args.random_state
            )

        # ── CV evaluation ────────────────────────────────────────────────────
        results = trainer.run_single_model(X, y, args.model_name)

        # ── Final model on all data ───────────────────────────────────────────
        monotonic_features = results['monotonicity_info']['monotonic_features']
        final_model, scaler, monotonic_features  = trainer.train_final_model(
            X, y, args.model_name, monotonic_features
        )


        # ── Persist ──────────────────────────────────────────────────────────
        save_results(results, args.output_folder, args.model_name)
        save_model(
            final_model, scaler, monotonic_features,
            args.model_name, args.output_folder,
            framework=args.framework,
        )

        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nOutput files in {args.output_folder}:")
        print("  - results.json")
        print("  - summary.txt")
        print("  - metrics.csv")
        print("  - model.pkl")
        if scaler is not None:
            print("  - scaler.pkl")
            print("  - model_metadata.json")
            print("  - features.txt")
        if args.framework == 'binary':
            print("  - pr_curve_folds.json     ← per-fold PR arrays")
            print("  - pr_curve_mean.json      ← interpolated mean ± std PR curve")
        else:
            print("  - scatter_folds.json      ← per-fold actual/predicted arrays")
            print("  - actual_vs_predicted.json← aggregated scatter + residuals")
        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())