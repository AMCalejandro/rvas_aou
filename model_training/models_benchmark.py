"""
Classifier & Regressor benchmark module with monotonicity constraints support.
Can be used for both batch benchmarking and single model training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LogisticRegression, Ridge, Lasso, ElasticNet
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve,   # ← ADD THIS
)
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
from enum import Enum
from collections import Counter

from sklearn.base import clone


# ---------------------------------------------------------------------------
# Shared enums & helpers
# ---------------------------------------------------------------------------

class MonotonicityType(Enum):
    """Types of monotonicity checking"""
    MARGINAL = "marginal"        # Spearman correlation between feature and target
    CONDITIONAL = "conditional"  # Model coefficients (linear models only)
    AUTO = "auto"                # Choose automatically based on model type


class MonotonicityEnforcer:
    """Check and enforce monotonicity constraints on features."""

    @staticmethod
    def check_monotonicity(X: pd.DataFrame, y_train: np.ndarray,
                           threshold: float = 0.0) -> Dict[str, float]:
        """
        Compute Spearman correlation between each feature and the target.
        Positive correlation → monotonic-increasing relationship.

        Returns:
            Dict mapping feature name → Spearman correlation.
        """
        correlations = {}
        for col in X.columns:
            mask = ~(pd.isna(X[col]) | pd.isna(y_train))
            if mask.sum() > 0:
                corr, _ = spearmanr(X[col][mask], y_train[mask])
                correlations[col] = corr
            else:
                correlations[col] = 0.0
        return correlations

    @staticmethod
    def check_coefficients(model, feature_names: List[str]) -> Dict[str, float]:
        """
        Extract coefficients from a fitted linear model.

        Returns:
            Dict mapping feature name → coefficient value.
        """
        if not hasattr(model, 'coef_'):
            raise ValueError("Model does not have a coef_ attribute.")

        coef_array = model.coef_
        if len(coef_array.shape) > 1:
            coef_array = coef_array[0]

        return {name: float(val) for name, val in zip(feature_names, coef_array)}

    @staticmethod
    def get_monotonic_features(scores: Dict[str, float],
                               threshold: float = 0.0) -> List[str]:
        """Return features whose score (correlation or coefficient) > threshold."""
        return [feat for feat, score in scores.items() if score > threshold]

    @staticmethod
    def supports_coefficient_check(model_name: str) -> bool:
        """True if the model exposes interpretable linear coefficients."""
        return model_name in {
            'Logistic Regression',
            'Ridge', 'Lasso', 'ElasticNet',
            'Linear SVM', 'Linear SVR',
            'SGD Classifier', 'Perceptron',
        }

    @staticmethod
    def iterative_monotonic_selection(
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        model,
        feature_names: List[str],
        needs_scaling: bool = False,
        threshold: float = 0.0,
    ) -> Tuple[List[str], object, Optional[StandardScaler]]:
        """
        Iteratively train a linear model, drop features with non-positive
        coefficients, and retrain — until all remaining coefficients are > threshold.

        Works identically for classifiers and regressors that expose coef_.

        Returns:
            (selected_features, trained_model, fitted_scaler_or_None)
        """
        current_features = list(feature_names)
        trained_model = None
        scaler = None

        for _ in range(len(feature_names) + 1):
            if not current_features:
                current_features = list(feature_names)
                break

            X_iter = X_train[current_features].copy()
            iter_model = clone(model)
            scaler = None

            if needs_scaling:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_iter)
                iter_model.fit(X_scaled, y_train)
            else:
                iter_model.fit(X_iter, y_train)

            trained_model = iter_model

            coefs = MonotonicityEnforcer.check_coefficients(iter_model, current_features)
            neg_features = [f for f, c in coefs.items() if c <= threshold]

            if not neg_features:
                break  # Converged — all coefficients are positive

            surviving = [f for f in current_features if f not in neg_features]
            if not surviving:
                break  # Removing all features is invalid — stop here

            current_features = surviving

        return current_features, trained_model, scaler


# ---------------------------------------------------------------------------
# Classifier Benchmark
# ---------------------------------------------------------------------------
class ClassifierBenchmark:
    """Benchmark multiple classifiers with proper handling of class imbalance
    and monotonicity constraints."""

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.results = []

    def get_monotone_constraints_string(self, feature_names: List[str]) -> str:
        return "(" + ",".join(["1"] * len(feature_names)) + ")"

    def get_models(self, feature_names: List[str]) -> Dict:
        monotone_str = self.get_monotone_constraints_string(feature_names)

        return {
            'XGBoost': XGBClassifier(
                max_depth=6, learning_rate=0.05, n_estimators=300,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, reg_alpha=0.0,
                monotone_constraints=monotone_str,
                eval_metric='logloss', random_state=self.random_state,
            ),
            'LightGBM': LGBMClassifier(
                max_depth=6, learning_rate=0.1, n_estimators=200,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                monotone_constraints=[1] * len(feature_names),
                random_state=self.random_state, verbose=-1,
            ),
            'XGBoost (Deep)': XGBClassifier(
                max_depth=8, learning_rate=0.03, n_estimators=400,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, reg_alpha=0.0,
                monotone_constraints=monotone_str,
                eval_metric='logloss', random_state=self.random_state,
            ),
            'LightGBM (Deep)': LGBMClassifier(
                max_depth=10, learning_rate=0.05, n_estimators=200,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                monotone_constraints=[1] * len(feature_names),
                random_state=self.random_state, verbose=-1,
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=self.random_state,
                penalty='l2', C=1.0,
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=400, max_depth=10,
                random_state=self.random_state, n_jobs=-1,
            ),
            'Linear SVM': LinearSVC(
                random_state=self.random_state, max_iter=5000,
            ),
            'Gaussian NB': GaussianNB(var_smoothing=1e-8),
        }

    def uses_native_monotonicity(self, model_name: str) -> bool:
        return model_name in {
            'XGBoost', 'LightGBM', 'XGBoost (Deep)', 'LightGBM (Deep)'
        }

    def evaluate_model_cv(self, X: pd.DataFrame, y: pd.Series,
                          model, model_name: str,
                          monotonicity_type: MonotonicityType = MonotonicityType.AUTO) -> Dict:
        """Evaluate a classifier via stratified cross-validation."""

        needs_scaling = model_name not in {
            'XGBoost', 'LightGBM', 'Random Forest',
            'XGBoost (Deep)', 'LightGBM (Deep)',
        }

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                              random_state=self.random_state)

        fold_metrics = {k: [] for k in [
            'avg_precision', 'roc_auc', 'f1_score',
            'precision', 'recall', 'tn', 'fp', 'fn', 'tp',
        ]}
        pr_curve_folds: List[Dict] = []

        print(f"\nEvaluating {model_name}")
        selected_features_across_folds = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            model_fold = clone(model)
            X_train = X.iloc[train_idx].copy()
            X_val   = X.iloc[val_idx].copy()
            y_train = y.iloc[train_idx]
            y_val   = y.iloc[val_idx]

            enforcer = MonotonicityEnforcer()

            if not self.uses_native_monotonicity(model_name):
                if monotonicity_type == MonotonicityType.AUTO:
                    check_type = (MonotonicityType.CONDITIONAL
                                  if enforcer.supports_coefficient_check(model_name)
                                  else MonotonicityType.MARGINAL)
                else:
                    check_type = monotonicity_type

                # ── Conditional: iterative coefficient pruning ──────────────
                if check_type == MonotonicityType.CONDITIONAL:
                    features_to_use, model_fold, scaler = \
                        MonotonicityEnforcer.iterative_monotonic_selection(
                            X_train, y_train, model_fold,
                            list(X_train.columns),
                            needs_scaling=needs_scaling, threshold=0.0,
                        )
                    selected_features_across_folds.append(set(features_to_use))

                    X_train_final = X_train[features_to_use].copy()
                    X_val_final   = X_val[features_to_use].copy()

                    if needs_scaling and scaler is not None:
                        X_train_final = scaler.transform(X_train_final)
                        X_val_final   = scaler.transform(X_val_final)
                    else:
                        X_train_final = X_train_final.values
                        X_val_final   = X_val_final.values

                # ── Marginal: Spearman-based filtering ──────────────────────
                else:
                    scores = enforcer.check_monotonicity(X_train, y_train)
                    features_to_use = enforcer.get_monotonic_features(scores, threshold=0.0)
                    selected_features_across_folds.append(set(features_to_use))

                    if not features_to_use:
                        features_to_use = list(X.columns)

                    X_train_final = X_train[features_to_use].copy()
                    X_val_final   = X_val[features_to_use].copy()

                    if needs_scaling:
                        scaler = StandardScaler()
                        X_train_final = scaler.fit_transform(X_train_final)
                        X_val_final   = scaler.transform(X_val_final)

                    model_fold.fit(X_train_final, y_train)

                # Linear SVM calibration (model_fold already trained inside iterative or marginal path)
                if model_name == "Linear SVM":
                    # NOTE: model_fold here is already the raw LinearSVC — do NOT use .estimator
                    calibrator = CalibratedClassifierCV(
                        estimator=model_fold,method="sigmoid",cv=3
                    )
                    calibrator.fit(X_train_final, y_train)
                    y_pred_proba = calibrator.predict_proba(X_val_final)[:, 1]
                    y_pred = calibrator.predict(X_val_final)
                else:
                    y_pred_proba = model_fold.predict_proba(X_val_final)[:, 1]
                    y_pred = model_fold.predict(X_val_final)
            
            else:
                features_to_use = list(X.columns)
                X_train_final = X_train.values
                X_val_final   = X_val.values
                model_fold.fit(X_train_final, y_train)
                y_pred_proba = model_fold.predict_proba(X_val_final)[:, 1]
                y_pred = model_fold.predict(X_val_final)

            avg_prec  = average_precision_score(y_val, y_pred_proba)
            roc_auc   = roc_auc_score(y_val, y_pred_proba)
            f1        = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall    = recall_score(y_val, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            # ── Collect PR curve data for this fold ─────────────────────────────
            prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_val, y_pred_proba)
            pr_curve_folds.append({
                'fold':      fold,
                'precision': prec_curve.tolist(),
                'recall':    rec_curve.tolist(),
                'threshold': thresh_curve.tolist(),   # len = len(prec_curve) - 1
            })

            fold_metrics['avg_precision'].append(avg_prec)
            fold_metrics['roc_auc'].append(roc_auc)
            fold_metrics['f1_score'].append(f1)
            fold_metrics['precision'].append(precision)
            fold_metrics['recall'].append(recall)
            fold_metrics['tn'].append(tn)
            fold_metrics['fp'].append(fp)
            fold_metrics['fn'].append(fn)
            fold_metrics['tp'].append(tp)

            print(f"  Fold {fold}: AP={avg_prec:.4f}, ROC-AUC={roc_auc:.4f}")

        # ── Aggregate monotonic features across folds ──────────────────────
        if not self.uses_native_monotonicity(model_name):
            feature_counter = Counter()
            for feat_set in selected_features_across_folds:
                feature_counter.update(feat_set)
            feature_frequency = {
                feat: count / self.n_folds
                for feat, count in feature_counter.items()
            }
            stable_monotonic_features = [
                feat for feat, freq in feature_frequency.items() if freq >= 0.5
            ]
        else:
            feature_frequency = {feat: 1.0 for feat in X.columns}
            stable_monotonic_features = list(X.columns)

        return {
            'model': model_name,
            'n_features_mean': np.mean([len(s) for s in selected_features_across_folds]),
            'n_features_std':  np.std([len(s) for s in selected_features_across_folds]),
            'monotonic_features':          stable_monotonic_features,
            'monotonic_feature_frequency': feature_frequency,
            'n_monotonic_features':        len(stable_monotonic_features),
            'uses_native_monotonicity':    self.uses_native_monotonicity(model_name),
            'avg_precision_mean': np.mean(fold_metrics['avg_precision']),
            'avg_precision_std':  np.std(fold_metrics['avg_precision']),
            'roc_auc_mean': np.mean(fold_metrics['roc_auc']),
            'roc_auc_std':  np.std(fold_metrics['roc_auc']),
            'f1_score_mean': np.mean(fold_metrics['f1_score']),
            'f1_score_std':  np.std(fold_metrics['f1_score']),
            'precision_mean': np.mean(fold_metrics['precision']),
            'precision_std':  np.std(fold_metrics['precision']),
            'recall_mean': np.mean(fold_metrics['recall']),
            'recall_std':  np.std(fold_metrics['recall']),
            'tn_mean': np.mean(fold_metrics['tn']),
            'fp_mean': np.mean(fold_metrics['fp']),
            'fn_mean': np.mean(fold_metrics['fn']),
            'tp_mean': np.mean(fold_metrics['tp']),
            'pr_curve_folds': pr_curve_folds,
        }

    def run_benchmark(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        feature_names = list(X.columns)
        models = self.get_models(feature_names)

        print("=" * 80)
        print("BINARY CLASSIFIER BENCHMARKING WITH MONOTONICITY CONSTRAINTS")
        print("=" * 80)
        print(f"\nDataset Information:")
        print(f"  Total samples:       {len(X)}")
        print(f"  Features:            {len(X.columns)}")
        print(f"  Positive class:      {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
        print(f"  Negative class:      {(~y.astype(bool)).sum()} ({(~y.astype(bool)).sum()/len(y)*100:.2f}%)")
        print(f"  CV folds:            {self.n_folds}")

        for model_name, model in models.items():
            start = time.time()
            print(f"\n{'='*80}\nMODEL: {model_name}\n{'='*80}")
            results = self.evaluate_model_cv(X, y, model, model_name)
            results['training_time'] = time.time() - start
            self.results.append(results)
            print(f"\n  Summary:")
            print(f"    Average Precision: {results['avg_precision_mean']:.4f} ± {results['avg_precision_std']:.4f}")
            print(f"    ROC-AUC:           {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
            print(f"    Training time:     {results['training_time']:.2f}s")

        results_df = pd.DataFrame(self.results)
        return results_df.sort_values('avg_precision_mean', ascending=False)


# ---------------------------------------------------------------------------
# Regressor Benchmark  (NEW)
# ---------------------------------------------------------------------------

class RegressorBenchmark:
    """
    Benchmark multiple regressors on a continuous target with monotonicity
    constraints enforced using the same strategy as ClassifierBenchmark:

      - XGBRegressor / LGBMRegressor  →  native monotone_constraints
      - Linear models (Ridge, Lasso, ElasticNet, LinearSVR)
                                      →  iterative coefficient pruning
      - Random Forest / GradientBoosting
                                      →  Spearman-based marginal filtering
    """

    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.results: List[Dict] = []

    # ── Model factory ────────────────────────────────────────────────────────

    def _monotone_str(self, feature_names: List[str]) -> str:
        return "(" + ",".join(["1"] * len(feature_names)) + ")"

    def get_models(self, feature_names: List[str]) -> Dict:
        """
        Return a dict of regression models.
        Tree-based models receive native monotone_constraints (all +1).
        Linear models' monotonicity is enforced post-hoc via coefficient pruning.
        """
        monotone_str = self._monotone_str(feature_names)

        return {
            # ── Tree-based with native constraints ──────────────────────────
            'XGBoost Regressor': XGBRegressor(
                max_depth=6, learning_rate=0.05, n_estimators=300,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, reg_alpha=0.0,
                monotone_constraints=monotone_str,
                eval_metric='rmse', random_state=self.random_state,
            ),
            'LightGBM Regressor': LGBMRegressor(
                max_depth=6, learning_rate=0.1, n_estimators=200,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                monotone_constraints=[1] * len(feature_names),
                random_state=self.random_state, verbose=-1,
            ),
            'XGBoost Regressor (Deep)': XGBRegressor(
                max_depth=8, learning_rate=0.03, n_estimators=400,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, reg_alpha=0.0,
                monotone_constraints=monotone_str,
                eval_metric='rmse', random_state=self.random_state,
            ),
            'LightGBM Regressor (Deep)': LGBMRegressor(
                max_depth=10, learning_rate=0.05, n_estimators=200,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                monotone_constraints=[1] * len(feature_names),
                random_state=self.random_state, verbose=-1,
            ),
            # ── Linear models — coefficient pruning applied post-hoc ────────
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.01, max_iter=5000),
            'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
            'Linear SVR': LinearSVR(max_iter=5000,
                                    random_state=self.random_state),
            # ── Ensemble — Spearman marginal filtering ──────────────────────
            'Random Forest Regressor': RandomForestRegressor(
                n_estimators=400, max_depth=10,
                random_state=self.random_state, n_jobs=-1,
            ),
            'Gradient Boosting Regressor': GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=self.random_state,
            ),
        }

    def uses_native_monotonicity(self, model_name: str) -> bool:
        return model_name in {
            'XGBoost Regressor', 'LightGBM Regressor',
            'XGBoost Regressor (Deep)', 'LightGBM Regressor (Deep)',
        }

    def _needs_scaling(self, model_name: str) -> bool:
        """Linear and SVR models benefit from feature scaling."""
        return model_name in {
            'Ridge', 'Lasso', 'ElasticNet', 'Linear SVR',
        }

    # ── Core CV evaluation ───────────────────────────────────────────────────

    def evaluate_model_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        model_name: str,
        monotonicity_type: MonotonicityType = MonotonicityType.AUTO,
    ) -> Dict:
        """
        Evaluate a regressor via K-Fold cross-validation and collect:
          - RMSE, MAE, R², Spearman ρ (between predictions and actuals)
          - Monotonic feature set (stable across folds)
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=self.random_state)

        fold_metrics: Dict[str, List] = {
            'rmse': [], 'mae': [], 'r2': [], 'spearman': [],
        }
        scatter_folds: List[Dict] = [] 

        print(f"\nEvaluating {model_name}")
        selected_features_across_folds: List[set] = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            model_fold = clone(model)
            X_train = X.iloc[train_idx].copy()
            X_val   = X.iloc[val_idx].copy()
            y_train = y.iloc[train_idx]
            y_val   = y.iloc[val_idx]

            needs_scaling = self._needs_scaling(model_name)
            enforcer = MonotonicityEnforcer()

            # ── Non-native: apply post-hoc monotonicity filtering ───────────
            if not self.uses_native_monotonicity(model_name):

                if monotonicity_type == MonotonicityType.AUTO:
                    check_type = (
                        MonotonicityType.CONDITIONAL
                        if enforcer.supports_coefficient_check(model_name)
                        else MonotonicityType.MARGINAL
                    )
                else:
                    check_type = monotonicity_type

                # ── Conditional: iterative coefficient pruning ──────────────
                if check_type == MonotonicityType.CONDITIONAL:
                    features_to_use, model_fold, scaler = \
                        MonotonicityEnforcer.iterative_monotonic_selection(
                            X_train, y_train, model_fold,
                            list(X_train.columns),
                            needs_scaling=needs_scaling,
                            threshold=0.0,
                        )
                    selected_features_across_folds.append(set(features_to_use))

                    X_train_final = X_train[features_to_use].copy()
                    X_val_final   = X_val[features_to_use].copy()

                    if needs_scaling and scaler is not None:
                        # scaler was fit inside iterative_monotonic_selection
                        X_train_final = scaler.transform(X_train_final)
                        X_val_final   = scaler.transform(X_val_final)
                    else:
                        X_train_final = X_train_final.values
                        X_val_final   = X_val_final.values

                    # model_fold is already trained inside iterative selection
                    y_pred = model_fold.predict(X_val_final)

                # ── Marginal: Spearman-based filtering ──────────────────────
                else:
                    scores = enforcer.check_monotonicity(X_train, y_train)
                    features_to_use = enforcer.get_monotonic_features(
                        scores, threshold=0.0
                    )
                    if not features_to_use:
                        features_to_use = list(X.columns)

                    selected_features_across_folds.append(set(features_to_use))

                    X_train_final = X_train[features_to_use].copy()
                    X_val_final   = X_val[features_to_use].copy()

                    if needs_scaling:
                        scaler = StandardScaler()
                        X_train_final = scaler.fit_transform(X_train_final)
                        X_val_final   = scaler.transform(X_val_final)
                    else:
                        X_train_final = X_train_final.values
                        X_val_final   = X_val_final.values

                    model_fold.fit(X_train_final, y_train)
                    y_pred = model_fold.predict(X_val_final)

            # ── Native monotonicity (XGBoost / LightGBM) ────────────────────
            else:
                features_to_use = list(X.columns)
                X_train_final = X_train.values
                X_val_final   = X_val.values
                model_fold.fit(X_train_final, y_train)
                y_pred = model_fold.predict(X_val_final)

            # ── Metrics ─────────────────────────────────────────────────────
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae  = mean_absolute_error(y_val, y_pred)
            r2   = r2_score(y_val, y_pred)
            sp_corr, _ = spearmanr(y_val, y_pred)

            scatter_folds.append({
                'fold':      fold,
                'y_true':    y_val.tolist(),
                'y_pred':    y_pred.tolist(),
            })

            fold_metrics['rmse'].append(rmse)
            fold_metrics['mae'].append(mae)
            fold_metrics['r2'].append(r2)
            fold_metrics['spearman'].append(sp_corr)

            print(f"  Fold {fold}: RMSE={rmse:.4f}, MAE={mae:.4f}, "
                  f"R²={r2:.4f}, Spearman ρ={sp_corr:.4f}")

        # ── Aggregate monotonic features across folds ────────────────────────
        if not self.uses_native_monotonicity(model_name):
            feature_counter: Counter = Counter()
            for feat_set in selected_features_across_folds:
                feature_counter.update(feat_set)
            feature_frequency = {
                feat: count / self.n_folds
                for feat, count in feature_counter.items()
            }
            stable_monotonic_features = [
                feat for feat, freq in feature_frequency.items()
                if freq >= 0.5
            ]
        else:
            feature_frequency = {feat: 1.0 for feat in X.columns}
            stable_monotonic_features = list(X.columns)

        return {
            'model': model_name,
            # Feature selection stats
            'n_features_mean': np.mean([len(s) for s in selected_features_across_folds]),
            'n_features_std':  np.std([len(s) for s in selected_features_across_folds]),
            'monotonic_features':          stable_monotonic_features,
            'monotonic_feature_frequency': feature_frequency,
            'n_monotonic_features':        len(stable_monotonic_features),
            'uses_native_monotonicity':    self.uses_native_monotonicity(model_name),
            # Regression metrics
            'rmse_mean':     np.mean(fold_metrics['rmse']),
            'rmse_std':      np.std(fold_metrics['rmse']),
            'mae_mean':      np.mean(fold_metrics['mae']),
            'mae_std':       np.std(fold_metrics['mae']),
            'r2_mean':       np.mean(fold_metrics['r2']),
            'r2_std':        np.std(fold_metrics['r2']),
            'spearman_mean': np.mean(fold_metrics['spearman']),
            'spearman_std':  np.std(fold_metrics['spearman']),
            'scatter_folds': scatter_folds,
        }

    # ── Pipeline entry-point ─────────────────────────────────────────────────

    def run_benchmark(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Run the full regression benchmark pipeline.

        Args:
            X: Feature matrix (all numeric, no NaN preferred).
            y: Continuous target vector.

        Returns:
            DataFrame with per-model CV metrics, sorted by RMSE ascending.
        """
        feature_names = list(X.columns)
        models = self.get_models(feature_names)

        print("=" * 80)
        print("REGRESSOR BENCHMARKING WITH MONOTONICITY CONSTRAINTS")
        print("=" * 80)
        print(f"\nDataset Information:")
        print(f"  Total samples:  {len(X)}")
        print(f"  Features:       {len(X.columns)}")
        print(f"  Target — mean:  {y.mean():.4f}")
        print(f"  Target — std:   {y.std():.4f}")
        print(f"  Target — range: [{y.min():.4f}, {y.max():.4f}]")
        print(f"  CV folds:       {self.n_folds}")
        print(f"\nMonotonicity Enforcement:")
        print(f"  - XGBoost/LightGBM: native monotone_constraints (all +1)")
        print(f"  - Linear models:    iterative coefficient pruning")
        print(f"  - Ensemble/other:   Spearman marginal feature filtering")

        for model_name, model in models.items():
            start = time.time()
            print(f"\n{'='*80}\nMODEL: {model_name}\n{'='*80}")

            results = self.evaluate_model_cv(X, y, model, model_name)
            results['training_time'] = time.time() - start
            self.results.append(results)

            print(f"\n  Summary:")
            print(f"    RMSE:         {results['rmse_mean']:.4f} ± {results['rmse_std']:.4f}")
            print(f"    MAE:          {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
            print(f"    R²:           {results['r2_mean']:.4f} ± {results['r2_std']:.4f}")
            print(f"    Spearman ρ:   {results['spearman_mean']:.4f} ± {results['spearman_std']:.4f}")
            print(f"    Training time:{results['training_time']:.2f}s")

        results_df = pd.DataFrame(self.results)
        # Primary sort: lowest RMSE first; secondary: highest R²
        return results_df.sort_values(
            ['rmse_mean', 'r2_mean'], ascending=[True, False]
        ).reset_index(drop=True)