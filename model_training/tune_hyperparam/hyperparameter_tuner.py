"""
Hyperparameter tuning module for classifier models.
Uses Optuna (Bayesian optimization) with cross-validation to tune
a single model while preventing overfitting through:
  - Pruning unpromising trials early
  - Regularization-aware search spaces
  - Train/val gap monitoring
  - Held-out test set for final unbiased evaluation
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import pickle
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from ..classifier_benchmark import (
    ClassifierBenchmark,
    MonotonicityEnforcer,
    MonotonicityType,
)
from ..run_model import SingleModelTrainer, load_data, load_predictors_from_file, save_model, save_results


# ---------------------------------------------------------------------------
# Search space definitions
# ---------------------------------------------------------------------------
def suggest_xgboost_params(trial: optuna.Trial, n_features: int) -> Dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def suggest_lightgbm_params(trial: optuna.Trial, n_features: int) -> Dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
    }


def suggest_logistic_params(trial: optuna.Trial, n_features: int) -> Dict:
    return {
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
        "solver": "saga",  # supports all penalties
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),  # only used for elasticnet
        "max_iter": 2000,
    }


def suggest_random_forest_params(trial: optuna.Trial, n_features: int) -> Dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "n_jobs": -1,
    }


def suggest_linear_svm_params(trial: optuna.Trial, n_features: int) -> Dict:
    return {
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "max_iter": 5000,
    }


def suggest_gaussian_nb_params(trial: optuna.Trial, n_features: int) -> Dict:
    return {
        "var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-3, log=True),
    }


SUGGEST_FN = {
    "XGBoost": suggest_xgboost_params,
    "XGBoost (Deep)": suggest_xgboost_params,
    "LightGBM": suggest_lightgbm_params,
    "LightGBM (Deep)": suggest_lightgbm_params,
    "Logistic Regression": suggest_logistic_params,
    "Random Forest": suggest_random_forest_params,
    "Linear SVM": suggest_linear_svm_params,
    "Gaussian NB": suggest_gaussian_nb_params,
}

NATIVE_MONO_MODELS = {"XGBoost", "XGBoost (Deep)", "LightGBM", "LightGBM (Deep)"}
NEEDS_SCALING = {
    "Logistic Regression",
    "Linear SVM",
    "Gaussian NB",
}


# ---------------------------------------------------------------------------
# Model builder: given name + params -> fresh sklearn estimator
# ---------------------------------------------------------------------------
def build_model(model_name: str, params: Dict, n_features: int,
                monotone_constraints=None, random_state: int = 42):
    """Instantiate a model from its name and a flat param dict."""

    if model_name in ("XGBoost", "XGBoost (Deep)"):
        mono = monotone_constraints or ("(" + ",".join(["1"] * n_features) + ")")
        return XGBClassifier(
            **params,
            monotone_constraints=mono,
            eval_metric="logloss",
            random_state=random_state,
        )

    if model_name in ("LightGBM", "LightGBM (Deep)"):
        mono = monotone_constraints or [1] * n_features
        return LGBMClassifier(
            **params,
            monotone_constraints=mono,
            random_state=random_state,
            verbose=-1,
        )

    if model_name == "Logistic Regression":
        # l1_ratio is only meaningful for elasticnet; passing it to other
        # solvers raises an error, so we guard it here.
        p = dict(params)
        if p.get("penalty") != "elasticnet":
            p.pop("l1_ratio", None)
        return LogisticRegression(**p, random_state=random_state)

    if model_name == "Random Forest":
        return RandomForestClassifier(**params, random_state=random_state)

    if model_name == "Linear SVM":
        return LinearSVC(**params, random_state=random_state)

    if model_name == "Gaussian NB":
        return GaussianNB(**params)

    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Overfitting guard: compute train-val AP gap across folds
# ---------------------------------------------------------------------------

def compute_train_val_gap(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    params: Dict,
    features_to_use: List[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Return (mean_val_ap, mean_train_ap, gap) where gap = train_ap - val_ap.
    Large gap → overfitting.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    X_feat = X[features_to_use]
    val_scores, train_scores = [], []

    for train_idx, val_idx in skf.split(X_feat, y):
        X_tr, X_vl = X_feat.iloc[train_idx].copy(), X_feat.iloc[val_idx].copy()
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

        mdl = build_model(model_name, params, len(features_to_use))
        needs_scaling = model_name in NEEDS_SCALING

        if needs_scaling:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_vl = scaler.transform(X_vl)

        if model_name == "Linear SVM":
            mdl.fit(X_tr, y_tr)
            cal = CalibratedClassifierCV(estimator=mdl, method="sigmoid", cv=3)
            cal.fit(X_tr, y_tr)
            val_scores.append(average_precision_score(y_vl, cal.predict_proba(X_vl)[:, 1]))
            train_scores.append(average_precision_score(y_tr, cal.predict_proba(X_tr)[:, 1]))
        else:
            mdl.fit(X_tr, y_tr)
            val_scores.append(average_precision_score(y_vl, mdl.predict_proba(X_vl)[:, 1]))
            train_scores.append(average_precision_score(y_tr, mdl.predict_proba(X_tr)[:, 1]))

    mean_val = float(np.mean(val_scores))
    mean_train = float(np.mean(train_scores))
    return mean_val, mean_train, mean_train - mean_val


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

class TuningObjective:
    """
    Optuna objective that:
      1. Proposes hyperparameters
      2. Runs stratified CV on the tuning split
      3. Reports intermediate fold scores for pruning
      4. Penalises large train-val gaps (overfitting)
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        features_to_use: List[str],
        n_folds: int = 5,
        random_state: int = 42,
        overfit_penalty: float = 0.5,
        max_overfit_gap: float = 0.15,
    ):
        self.X = X
        self.y = y
        self.model_name = model_name
        self.features_to_use = features_to_use
        self.n_folds = n_folds
        self.random_state = random_state
        self.overfit_penalty = overfit_penalty  # weight on gap penalty
        self.max_overfit_gap = max_overfit_gap  # hard cap: gap > this → prune

    def __call__(self, trial: optuna.Trial) -> float:
        n_features = len(self.features_to_use)
        params = SUGGEST_FN[self.model_name](trial, n_features)
        X_feat = self.X[self.features_to_use]
        needs_scaling = self.model_name in NEEDS_SCALING

        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        val_scores, train_scores = [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_feat, self.y)):
            X_tr = X_feat.iloc[train_idx].copy()
            X_vl = X_feat.iloc[val_idx].copy()
            y_tr = self.y.iloc[train_idx]
            y_vl = self.y.iloc[val_idx]

            mdl = build_model(self.model_name, params, n_features)

            if needs_scaling:
                scaler = StandardScaler()
                X_tr_arr = scaler.fit_transform(X_tr)
                X_vl_arr = scaler.transform(X_vl)
            else:
                X_tr_arr, X_vl_arr = X_tr, X_vl

            if self.model_name == "Linear SVM":
                mdl.fit(X_tr_arr, y_tr)
                cal = CalibratedClassifierCV(estimator=mdl, method="sigmoid", cv=3)
                cal.fit(X_tr_arr, y_tr)
                v_ap = average_precision_score(y_vl, cal.predict_proba(X_vl_arr)[:, 1])
                t_ap = average_precision_score(y_tr, cal.predict_proba(X_tr_arr)[:, 1])
            else:
                mdl.fit(X_tr_arr, y_tr)
                v_ap = average_precision_score(y_vl, mdl.predict_proba(X_vl_arr)[:, 1])
                t_ap = average_precision_score(y_tr, mdl.predict_proba(X_tr_arr)[:, 1])

            val_scores.append(v_ap)
            train_scores.append(t_ap)

            # Report intermediate value for pruning
            trial.report(float(np.mean(val_scores)), step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        mean_val = float(np.mean(val_scores))
        mean_train = float(np.mean(train_scores))
        gap = mean_train - mean_val

        # Hard prune: severe overfitting
        if gap > self.max_overfit_gap:
            raise optuna.exceptions.TrialPruned()

        # Soft penalty: penalise gap proportionally
        penalised_score = mean_val - self.overfit_penalty * max(gap, 0.0)
        return penalised_score


# ---------------------------------------------------------------------------
# Main tuner class
# ---------------------------------------------------------------------------

class HyperparameterTuner(SingleModelTrainer):
    """
    Extends SingleModelTrainer with Optuna-based hyperparameter tuning.

    Workflow
    --------
    1. Split data into tuning (80%) and held-out test (20%) sets.
    2. On the tuning set, run standard CV to identify monotonic features.
    3. Run Optuna study on the tuning set.
    4. Evaluate best params on held-out test set (unbiased).
    5. Train final model on ALL data with best params.
    6. Save model + results.
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        overfit_penalty: float = 0.5,
        max_overfit_gap: float = 0.15,
        test_size: float = 0.2,
    ):
        super().__init__(n_folds=n_folds, random_state=random_state)
        self.n_trials = n_trials
        self.timeout = timeout
        self.overfit_penalty = overfit_penalty
        self.max_overfit_gap = max_overfit_gap
        self.test_size = test_size

    # ------------------------------------------------------------------
    # Identify monotonic features (on the tuning split only)
    # ------------------------------------------------------------------

    def _identify_monotonic_features(
        self, X_tune: pd.DataFrame, y_tune: pd.Series, model_name: str
    ) -> List[str]:
        """Run one CV pass to identify stable monotonic features."""
        print("\n[Step 1] Identifying monotonic features via CV …")
        cv_results = self.evaluate_model_cv(X_tune, y_tune,
                                            self.get_models(list(X_tune.columns))[model_name],
                                            model_name)
        features = cv_results.get("monotonic_features", list(X_tune.columns))
        if not features:
            features = list(X_tune.columns)
        print(f"  → {len(features)}/{len(X_tune.columns)} monotonic features identified")
        return features, cv_results

    # ------------------------------------------------------------------
    # Run Optuna study
    # ------------------------------------------------------------------

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
    ) -> Dict:
        """
        Full tuning pipeline. Returns a results dict ready for save_results().
        """
        if model_name not in SUGGEST_FN:
            raise ValueError(
                f"No search space defined for '{model_name}'. "
                f"Available: {list(SUGGEST_FN.keys())}"
            )

        print("=" * 80)
        print(f"HYPERPARAMETER TUNING: {model_name}")
        print("=" * 80)
        print(f"\nDataset: {len(X)} samples, {len(X.columns)} features")
        print(f"Positive class: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        print(f"Trials: {self.n_trials}  |  Timeout: {self.timeout}s  |  CV folds: {self.n_folds}")
        print(f"Overfit penalty: {self.overfit_penalty}  |  Max gap: {self.max_overfit_gap}")

        # ----------------------------------------------------------------
        # 1. Train/test split (stratified, held-out set never seen by Optuna)
        # ----------------------------------------------------------------
        X_tune, X_test, y_tune, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )
        print(f"\nTuning set : {len(X_tune)} samples")
        print(f"Test set   : {len(X_test)} samples (held-out, never used during tuning)")

        # ----------------------------------------------------------------
        # 2. Monotonic feature selection (on tuning set)
        # ----------------------------------------------------------------
        monotonic_features, baseline_cv_results = self._identify_monotonic_features(
            X_tune, y_tune, model_name
        )

        # ----------------------------------------------------------------
        # 3. Optuna study
        # ----------------------------------------------------------------
        print(f"\n[Step 2] Running Optuna study ({self.n_trials} trials) …")
        t0 = time.time()

        sampler = TPESampler(seed=self.random_state, n_startup_trials=max(10, self.n_trials // 10))
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

        objective = TuningObjective(
            X=X_tune,
            y=y_tune,
            model_name=model_name,
            features_to_use=monotonic_features,
            n_folds=self.n_folds,
            random_state=self.random_state,
            overfit_penalty=self.overfit_penalty,
            max_overfit_gap=self.max_overfit_gap,
        )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
            n_jobs=1,
        )

        tuning_time = time.time() - t0
        best_trial = study.best_trial
        best_params = best_trial.params
        best_penalised_score = best_trial.value

        print(f"\n  Completed {len(study.trials)} trials in {tuning_time:.1f}s")
        print(f"  Best penalised AP (tuning CV): {best_penalised_score:.4f}")
        print(f"  Best params: {json.dumps(best_params, indent=4)}")

        # ----------------------------------------------------------------
        # 4. Re-evaluate best params with full gap diagnostics (tuning set)
        # ----------------------------------------------------------------
        print("\n[Step 3] Evaluating best params with train/val gap diagnostics …")
        mean_val_ap, mean_train_ap, gap = compute_train_val_gap(
            X_tune, y_tune, model_name, best_params, monotonic_features,
            n_folds=self.n_folds, random_state=self.random_state,
        )
        print(f"  Train AP : {mean_train_ap:.4f}")
        print(f"  Val   AP : {mean_val_ap:.4f}")
        print(f"  Gap      : {gap:.4f}  {'⚠ POSSIBLE OVERFIT' if gap > 0.05 else '✓ OK'}")

        # ----------------------------------------------------------------
        # 5. Held-out test evaluation
        # ----------------------------------------------------------------
        print("\n[Step 4] Evaluating on held-out test set …")
        test_ap, test_roc = self._evaluate_on_test(
            X_tune, y_tune, X_test, y_test,
            model_name, best_params, monotonic_features,
        )
        print(f"  Test AP     : {test_ap:.4f}")
        print(f"  Test ROC-AUC: {test_roc:.4f}")

        # ----------------------------------------------------------------
        # 6. Train final model on ALL data with best params
        # ----------------------------------------------------------------
        print("\n[Step 5] Training final model on ALL data with best params …")
        final_model, scaler = self._train_final_with_params(
            X, y, model_name, best_params, monotonic_features
        )

        # ----------------------------------------------------------------
        # Build result dict (compatible with save_results)
        # ----------------------------------------------------------------
        results = {
            "model_performance": {
                **baseline_cv_results,
                # Overwrite with tuned metrics
                "avg_precision_mean": mean_val_ap,
                "avg_precision_std": 0.0,
                "train_avg_precision_mean": mean_train_ap,
                "train_val_gap": gap,
                "test_avg_precision": test_ap,
                "test_roc_auc": test_roc,
                "tuning_time": tuning_time,
                "n_trials_completed": len(study.trials),
                "n_trials_pruned": sum(
                    1 for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED
                ),
                "best_penalised_score": best_penalised_score,
                "best_params": best_params,
            },
            "dataset_info": {
                "n_samples": len(X),
                "n_features": len(X.columns),
                "feature_names": list(X.columns),
                "n_positive": int(y.sum()),
                "n_negative": int((~y.astype(bool)).sum()),
                "positive_rate": float(y.sum() / len(y)),
                "n_tune_samples": len(X_tune),
                "n_test_samples": len(X_test),
            },
            "monotonicity_info": {
                "uses_native_monotonicity": model_name in NATIVE_MONO_MODELS,
                "n_monotonic_features": len(monotonic_features),
                "monotonic_features": monotonic_features,
                "monotonic_feature_frequency": baseline_cv_results.get(
                    "monotonic_feature_frequency", {}
                ),
                "n_features_mean": baseline_cv_results.get("n_features_mean", len(monotonic_features)),
                "n_features_std": baseline_cv_results.get("n_features_std", 0.0),
            },
            "tuning_config": {
                "n_trials": self.n_trials,
                "timeout": self.timeout,
                "overfit_penalty": self.overfit_penalty,
                "max_overfit_gap": self.max_overfit_gap,
                "test_size": self.test_size,
            },
        }

        return results, final_model, scaler, monotonic_features

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate_on_test(
        self,
        X_tune: pd.DataFrame,
        y_tune: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str,
        params: Dict,
        features_to_use: List[str],
    ) -> Tuple[float, float]:
        """Train on tune set with best params, evaluate on test set."""
        n_features = len(features_to_use)
        mdl = build_model(model_name, params, n_features)
        needs_scaling = model_name in NEEDS_SCALING

        X_tr = X_tune[features_to_use].copy()
        X_vl = X_test[features_to_use].copy()

        if needs_scaling:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_vl = scaler.transform(X_vl)

        if model_name == "Linear SVM":
            mdl.fit(X_tr, y_tune)
            cal = CalibratedClassifierCV(estimator=mdl, method="sigmoid", cv=3)
            cal.fit(X_tr, y_tune)
            proba = cal.predict_proba(X_vl)[:, 1]
        else:
            mdl.fit(X_tr, y_tune)
            proba = mdl.predict_proba(X_vl)[:, 1]

        return (
            average_precision_score(y_test, proba),
            roc_auc_score(y_test, proba),
        )

    def _train_final_with_params(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        params: Dict,
        features_to_use: List[str],
    ) -> Tuple[object, Optional[StandardScaler]]:
        """Train on the full dataset using tuned params."""
        n_features = len(features_to_use)
        mdl = build_model(model_name, params, n_features)
        needs_scaling = model_name in NEEDS_SCALING

        X_feat = X[features_to_use].copy()
        scaler = None

        if needs_scaling:
            scaler = StandardScaler()
            X_arr = scaler.fit_transform(X_feat)
        else:
            X_arr = X_feat

        if model_name == "Linear SVM":
            mdl.fit(X_arr, y)
            cal = CalibratedClassifierCV(estimator=mdl, method="sigmoid", cv=3)
            cal.fit(X_arr, y)
            return cal, scaler

        mdl.fit(X_arr, y)
        return mdl, scaler


# ---------------------------------------------------------------------------
# save_tuning_summary – extends save_results with tuning-specific info
# ---------------------------------------------------------------------------

def save_tuning_summary(results: Dict, output_folder: str, model_name: str):
    """Saves JSON, CSV, and a human-readable summary for a tuning run."""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- JSON ---
    json_path = output_path / "results.json"

    def _convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(i) for i in obj]
        return obj

    with open(json_path, "w") as f:
        json.dump(_convert(results), f, indent=2)
    print(f"Results saved → {json_path}")

    # --- Summary TXT ---
    summary_path = output_path / "summary.txt"
    perf = results["model_performance"]
    mono = results["monotonicity_info"]
    data = results["dataset_info"]
    cfg = results["tuning_config"]

    with open(summary_path, "w") as f:
        w = f.write
        w("=" * 80 + "\n")
        w(f"HYPERPARAMETER TUNING RESULTS: {model_name}\n")
        w("=" * 80 + "\n\n")

        w("DATASET\n" + "-" * 40 + "\n")
        w(f"  Total samples : {data['n_samples']}\n")
        w(f"  Tuning set    : {data['n_tune_samples']}\n")
        w(f"  Test set      : {data['n_test_samples']} (held-out)\n")
        w(f"  Positive rate : {data['positive_rate']*100:.2f}%\n\n")

        w("TUNING CONFIG\n" + "-" * 40 + "\n")
        w(f"  Trials requested      : {cfg['n_trials']}\n")
        w(f"  Trials completed      : {perf['n_trials_completed']}\n")
        w(f"  Trials pruned         : {perf['n_trials_pruned']}\n")
        w(f"  Overfit penalty       : {cfg['overfit_penalty']}\n")
        w(f"  Max overfit gap       : {cfg['max_overfit_gap']}\n")
        w(f"  Total tuning time     : {perf['tuning_time']:.1f}s\n\n")

        w("BEST HYPERPARAMETERS\n" + "-" * 40 + "\n")
        for k, v in perf["best_params"].items():
            w(f"  {k:<30} {v}\n")
        w("\n")

        w("PERFORMANCE (best params)\n" + "-" * 40 + "\n")
        w(f"  Train AP (CV mean)    : {perf.get('train_avg_precision_mean', float('nan')):.4f}\n")
        w(f"  Val   AP (CV mean)    : {perf['avg_precision_mean']:.4f}\n")
        w(f"  Train-Val Gap         : {perf.get('train_val_gap', float('nan')):.4f}"
          f"  {'⚠ POSSIBLE OVERFIT' if perf.get('train_val_gap', 0) > 0.05 else '✓ OK'}\n")
        w(f"  Test  AP (held-out)   : {perf.get('test_avg_precision', float('nan')):.4f}\n")
        w(f"  Test ROC-AUC          : {perf.get('test_roc_auc', float('nan')):.4f}\n\n")

        w("MONOTONICITY\n" + "-" * 40 + "\n")
        w(f"  Native monotonicity   : {mono['uses_native_monotonicity']}\n")
        w(f"  Monotonic features    : {mono['n_monotonic_features']}/{data['n_features']}\n")
        for feat in sorted(mono["monotonic_features"]):
            freq = mono["monotonic_feature_frequency"].get(feat, 1.0)
            w(f"    - {feat} ({freq*100:.0f}% of folds)\n")

        excluded = set(data["feature_names"]) - set(mono["monotonic_features"])
        if excluded:
            w(f"\n  Excluded features:\n")
            for feat in sorted(excluded):
                freq = mono["monotonic_feature_frequency"].get(feat, 0.0)
                w(f"    - {feat} ({freq*100:.0f}% of folds)\n")

    print(f"Summary saved → {summary_path}")

    # --- CSV ---
    csv_path = output_path / "metrics.csv"
    pd.DataFrame([{
        "model": model_name,
        "val_avg_precision": perf["avg_precision_mean"],
        "train_avg_precision": perf.get("train_avg_precision_mean", float("nan")),
        "train_val_gap": perf.get("train_val_gap", float("nan")),
        "test_avg_precision": perf.get("test_avg_precision", float("nan")),
        "test_roc_auc": perf.get("test_roc_auc", float("nan")),
        "n_trials_completed": perf["n_trials_completed"],
        "n_trials_pruned": perf["n_trials_pruned"],
        "tuning_time_s": perf["tuning_time"],
        "n_monotonic_features": mono["n_monotonic_features"],
        **{f"param_{k}": v for k, v in perf["best_params"].items()},
    }]).to_csv(csv_path, index=False)
    print(f"Metrics CSV saved → {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for a single classifier with monotonicity constraints"
    )
    parser.add_argument("model_name", type=str, help="Model to tune (e.g. XGBoost, LightGBM, 'Logistic Regression')")
    parser.add_argument("input_data", type=str, help="Path to CSV/TSV input file")
    parser.add_argument("output_folder", type=str, help="Path to output folder")

    parser.add_argument("--target-column", type=str, default="target")
    parser.add_argument("--predictors", type=str, nargs="+", required=False)
    parser.add_argument("--predictors-file", type=str, required=False)
    parser.add_argument("--framework", type=str, default="binary")

    thr = parser.add_mutually_exclusive_group()
    thr.add_argument("--bin-threshold", type=float, default=None)
    thr.add_argument("--top-percent", type=float, default=None)

    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sep", type=str, default=",")

    # Tuning-specific
    parser.add_argument("--n-trials", type=int, default=1500, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=3500, help="Max tuning time in seconds")
    parser.add_argument("--overfit-penalty", type=float, default=0.5,
                        help="Penalty weight on train-val gap (default 0.5)")
    parser.add_argument("--max-overfit-gap", type=float, default=0.15,
                        help="Hard gap limit above which trial is pruned (default 0.15)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction held out as test set (default 0.2)")

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 80)
    print("HYPERPARAMETER TUNING SCRIPT")
    print("=" * 80)
    print(f"\nModel        : {args.model_name}")
    print(f"Input        : {args.input_data}")
    print(f"Output       : {args.output_folder}")
    print(f"Trials       : {args.n_trials}")
    print(f"Timeout      : {args.timeout}s")
    print(f"Overfit pen  : {args.overfit_penalty}")
    print(f"Max gap      : {args.max_overfit_gap}")
    print(f"Test size    : {args.test_size}")

    try:
        predictors = (
            load_predictors_from_file(args.predictors_file)
            if args.predictors_file
            else args.predictors
        )

        X, y = load_data(
            args.input_data,
            args.target_column,
            predictors,
            args.framework,
            args.bin_threshold,
            args.top_percent,
            args.sep,
        )

        tuner = HyperparameterTuner(
            n_folds=args.n_folds,
            random_state=args.random_state,
            n_trials=args.n_trials,
            timeout=args.timeout,
            overfit_penalty=args.overfit_penalty,
            max_overfit_gap=args.max_overfit_gap,
            test_size=args.test_size,
        )

        results, final_model, scaler, monotonic_features = tuner.tune(X, y, args.model_name)

        save_tuning_summary(results, args.output_folder, args.model_name)
        save_model(final_model, scaler, monotonic_features, args.model_name, args.output_folder)

        print("\n" + "=" * 80)
        print("TUNING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nOutputs in {args.output_folder}:")
        print("  tuning_results.json   – full results + all trial details")
        print("  tuning_summary.txt    – human-readable summary")
        print("  tuning_metrics.csv    – key metrics + best params")
        print("  model.pkl             – final trained model (all data)")
        if scaler is not None:
            print("  scaler.pkl            – fitted StandardScaler")
        print("  model_metadata.json   – features + model info")
        print("  features.txt          – monotonic feature list")
        return 0

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())