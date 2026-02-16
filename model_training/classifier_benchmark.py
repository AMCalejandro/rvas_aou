"""
Classifier benchmark module with monotonicity constraints support.
Can be used for both batch benchmarking and single model training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, 
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
from enum import Enum

from sklearn.base import clone


class MonotonicityType(Enum):
    """Types of monotonicity checking"""
    MARGINAL = "marginal"      # Check correlation between feature and predictions
    CONDITIONAL = "conditional"  # Check model coefficients (for linear models)
    AUTO = "auto"              # Choose automatically based on model type

class MonotonicityEnforcer:
    """Check and enforce monotonicity constraints on features"""
    
    @staticmethod
    def check_monotonicity(X: pd.DataFrame, y_train: np.ndarray, 
                          threshold: float = 0.0) -> Dict[str, float]:
        """
        Check Spearman correlation between features and predicted probabilities.
        Positive correlation indicates monotonic increasing relationship.
        
        Args:
            X: Feature dataframe
            y_pred_proba: Predicted probabilities
            threshold: Minimum absolute correlation to consider monotonic
            
        Returns:
            Dictionary of feature names and their correlations
        """
        correlations = {}
        for col in X.columns:
            # Remove NaN values for correlation calculation
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
        Check model coefficients for conditional monotonicity.
        Works for linear models (LogisticRegression, LinearSVC, etc.)
        
        Args:
            model: Fitted model with coef_ attribute
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature names and their coefficients
        """
        if not hasattr(model, 'coef_'):
            raise ValueError("Model does not have coefficients (coef_ attribute)")
        
        coefficients = {}
        coef_array = model.coef_
        
        # Handle binary classification (shape could be (1, n_features) or (n_features,))
        if len(coef_array.shape) > 1:
            coef_array = coef_array[0]
        
        for feature_name, coef_value in zip(feature_names, coef_array):
            coefficients[feature_name] = float(coef_value)
        
        return coefficients
    
    @staticmethod
    def get_monotonic_features(scores: Dict[str, float], 
                              threshold: float = 0.0) -> List[str]:
        """
        Filter features that show monotonic relationship.
        
        Args:
            scores: Dictionary of feature scores (correlations or coefficients)
            threshold: Minimum score to keep feature
            
        Returns:
            List of monotonic feature names
        """
        # Keep features with positive scores (monotonic increasing)
        monotonic_features = [
            feat for feat, score in scores.items() 
            if score > threshold
        ]
        return monotonic_features
    
    @staticmethod
    def supports_coefficient_check(model_name: str) -> bool:
        """
        Check if model type supports coefficient-based monotonicity checking.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model has interpretable coefficients
        """
        coefficient_models = {
            'Logistic Regression',
            'Ridge',
            'Lasso',
            'ElasticNet',
            'Linear SVM',
            'SGD Classifier',
            'Perceptron'
        }
        return model_name in coefficient_models

class ClassifierBenchmark:
    """Benchmark multiple classifiers with proper handling of class imbalance"""
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.results = []
        self.monotonicity_results = []
        
    def get_monotone_constraints_string(self, feature_names: List[str]) -> str:
        """
        Generate monotone_constraints string for XGBoost/LightGBM.
        All features constrained to be monotonically increasing (1).
        
        Args:
            feature_names: List of feature names
            
        Returns:
            String like "(1,1,1,...,1)" for all features
        """
        return "(" + ",".join(["1"] * len(feature_names)) + ")"
    
    def get_models(self, feature_names: List[str]) -> Dict:
        """
        Define models with class imbalance handling and monotonicity constraints.
        
        Args:
            feature_names: List of feature names for monotone_constraints
            
        Returns:
            Dictionary of model name -> model instance
        """
        
        # Generate monotonic constraints for tree-based models
        monotone_constraints_str = self.get_monotone_constraints_string(feature_names)
        
        models = {
            # Tree-based models WITH native monotonic constraints
            'XGBoost': XGBClassifier(
                max_depth=6,
                learning_rate=0.05,
                n_estimators=300,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                reg_alpha=0.0,
                monotone_constraints=monotone_constraints_str,
                eval_metric='logloss',
                random_state=self.random_state
            ),
            'LightGBM': LGBMClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                monotone_constraints=[1] * len(feature_names),
                random_state=self.random_state,
                verbose=-1
            ),
            'XGBoost (Deep)': XGBClassifier(
                max_depth=8,
                learning_rate=0.03,
                n_estimators=400,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                reg_alpha=0.0,
                monotone_constraints=monotone_constraints_str,
                eval_metric='logloss',
                random_state=self.random_state
            ),
            'LightGBM (Deep)': LGBMClassifier(
                max_depth=10,
                learning_rate=0.05,
                n_estimators=200,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                monotone_constraints=[1] * len(feature_names),  # NATIVE MONOTONICITY
                random_state=self.random_state,
                verbose=-1
            ),
            # Non-tree models (will use post-hoc feature filtering)
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state,
                # solver='lbfgs',
                penalty='l2',
                C=1.0
            ),
            'Random Forest': RandomForestClassifier(
                class_weight='balanced',
               n_estimators=400,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Linear SVM': LinearSVC(
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=5000
            ),
            'Gaussian NB': GaussianNB(
                var_smoothing=1e-8
            )
        }
        
        return models
    
    def uses_native_monotonicity(self, model_name: str) -> bool:
        """
        Check if model uses native monotonicity constraints.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model has native monotonicity support
        """
        native_monotonicity_models = ['XGBoost', 'LightGBM', 'XGBoost (Deep)', 'LightGBM (Deep)']
        return model_name in native_monotonicity_models
    
    def evaluate_model_cv(self, X: pd.DataFrame, y: pd.Series, 
                         model, model_name: str,
                         monotonicity_type: MonotonicityType = MonotonicityType.AUTO) -> Dict:
        """
        Evaluate model using stratified cross-validation.
        
        Args:
            X: Feature matrix (will be filtered to features_to_use)
            y: Target vector
            model: Sklearn-compatible model
            model_name: Name of the model
            features_to_use: List of features to use
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        # Check if model needs scaling
        needs_scaling = model_name not in ['XGBoost', 'LightGBM', 'Random Forest',
                                            'XGBoost (Deep)', 'LightGBM (Deep)']
        
        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                              random_state=self.random_state)
        
        # Store metrics across folds
        fold_metrics = {
            'avg_precision': [],
            'roc_auc': [],
            'f1_score': [],
            'precision': [],
            'recall': [],
            'tn': [],
            'fp': [],
            'fn': [],
            'tp': []
        }
        
        print(f"\nEvaluating {model_name}")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            model_fold = clone(model)

            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            # --------------------------------------------------
            # Step 1: Handle imbalance for tree models
            # --------------------------------------------------
            n_pos = np.sum(y_train == 1)
            n_neg = np.sum(y_train == 0)
            scale_pos_weight = n_neg / max(n_pos, 1)

            if model_name in ['XGBoost', 'XGBoost (Deep)']:
                model_fold.set_params(scale_pos_weight=scale_pos_weight)

            if model_name in ['LightGBM', 'LightGBM (Deep)']:
                model_fold.set_params(scale_pos_weight=scale_pos_weight)

            # --------------------------------------------------
            # Step 2: Nested Monotonic Feature Selection
            # --------------------------------------------------
            enforcer = MonotonicityEnforcer()

            # Store selected features per fold
            if fold == 1:
                selected_features_across_folds = []

            features_to_use = list(X.columns)

            if not self.uses_native_monotonicity(model_name):

                # Decide monotonicity check type
                if monotonicity_type == MonotonicityType.AUTO:
                    if enforcer.supports_coefficient_check(model_name):
                        check_type = MonotonicityType.CONDITIONAL
                    else:
                        check_type = MonotonicityType.MARGINAL
                else:
                    check_type = monotonicity_type

                temp_model = clone(model)

                # Scaling if needed
                needs_scaling = model_name not in [
                    'XGBoost', 'LightGBM',
                    'Random Forest',
                    'XGBoost (Deep)', 'LightGBM (Deep)'
                ]

                if needs_scaling:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    temp_model.fit(X_train_scaled, y_train)
                else:
                    temp_model.fit(X_train, y_train)

                # ----------------------------
                # CONDITIONAL MONOTONICITY
                # ----------------------------
                if check_type == MonotonicityType.CONDITIONAL:
                    try:
                        scores = enforcer.check_coefficients(
                            temp_model.estimator if model_name == "Linear SVM" else temp_model,
                            list(X.columns)
                        )
                    except Exception:
                        # fallback
                        # y_train_pred = temp_model.predict_proba(
                        #     X_train_scaled if needs_scaling else X_train
                        # )[:, 1]
                        scores = enforcer.check_monotonicity(X_train, y_train)

                # ----------------------------
                # MARGINAL MONOTONICITY
                # ----------------------------
                else:
                    # y_train_pred = temp_model.predict_proba(
                    #     X_train_scaled if needs_scaling else X_train
                    # )[:, 1]

                    scores = enforcer.check_monotonicity(X_train, y_train)

                features_to_use = enforcer.get_monotonic_features(scores, threshold=0.0)

                if len(features_to_use) == 0:
                    features_to_use = list(X.columns)

            # Track fold-level selection
            selected_features_across_folds.append(set(features_to_use))

            # Apply feature filtering
            X_train = X_train[features_to_use]
            X_val = X_val[features_to_use]

            # --------------------------------------------------
            # Step 3: Final Training
            # --------------------------------------------------
            needs_scaling = model_name not in [
                'XGBoost', 'LightGBM',
                'Random Forest',
                'XGBoost (Deep)', 'LightGBM (Deep)'
            ]

            if needs_scaling:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            if model_name == "Linear SVM":
                # Fit base SVM
                model_fold.fit(X_train, y_train)

                # Calibrate on training data (modern sklearn)
                calibrator = CalibratedClassifierCV(
                    estimator=model_fold,
                    method="sigmoid",
                    cv=3  # Proper nested calibration
                )
                calibrator.fit(X_train, y_train)
                y_pred_proba = calibrator.predict_proba(X_val)[:, 1]
                y_pred = calibrator.predict(X_val)
            else:
                model_fold.fit(X_train, y_train)
                y_pred_proba = model_fold.predict_proba(X_val)[:, 1]
                y_pred = model_fold.predict(X_val)

            # --------------------------------------------------
            # Step 4: Metrics
            # --------------------------------------------------
            avg_prec = average_precision_score(y_val, y_pred_proba)
            roc_auc = roc_auc_score(y_val, y_pred_proba)
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

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
        
        # --------------------------------------------------
        # Aggregate monotonic features across folds
        # --------------------------------------------------

        if not self.uses_native_monotonicity(model_name):

            from collections import Counter

            feature_counter = Counter()
            for feat_set in selected_features_across_folds:
                for feat in feat_set:
                    feature_counter[feat] += 1

            # Frequency of selection
            feature_frequency = {
                feat: count / self.n_folds
                for feat, count in feature_counter.items()
            }

            # Stable monotonic features (selected in >=50% folds)
            stable_monotonic_features = [
                feat for feat, freq in feature_frequency.items()
                if freq >= 0.5
            ]

        else:
            feature_frequency = {feat: 1.0 for feat in X.columns}
            stable_monotonic_features = list(X.columns)

        # Calculate mean and std across folds
        results = {
            'model': model_name,
            'n_features_mean': np.mean([len(s) for s in selected_features_across_folds]),
            'n_features_std': np.std([len(s) for s in selected_features_across_folds]),
            'monotonic_features': stable_monotonic_features,
            'monotonic_feature_frequency': feature_frequency,
            'n_monotonic_features': len(stable_monotonic_features),
            'uses_native_monotonicity': self.uses_native_monotonicity(model_name),
            'avg_precision_mean': np.mean(fold_metrics['avg_precision']),
            'avg_precision_std': np.std(fold_metrics['avg_precision']),
            'roc_auc_mean': np.mean(fold_metrics['roc_auc']),
            'roc_auc_std': np.std(fold_metrics['roc_auc']),
            'f1_score_mean': np.mean(fold_metrics['f1_score']),
            'f1_score_std': np.std(fold_metrics['f1_score']),
            'precision_mean': np.mean(fold_metrics['precision']),
            'precision_std': np.std(fold_metrics['precision']),
            'recall_mean': np.mean(fold_metrics['recall']),
            'recall_std': np.std(fold_metrics['recall']),
            'tn_mean': np.mean(fold_metrics['tn']),
            'fp_mean': np.mean(fold_metrics['fp']),
            'fn_mean': np.mean(fold_metrics['fn']),
            'tp_mean': np.mean(fold_metrics['tp']),
        }
        
        return results
    
    def run_benchmark(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Run complete benchmark pipeline for all models.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            DataFrame with results for all models
        """
        feature_names = list(X.columns)
        models = self.get_models(feature_names)
        
        print("="*80)
        print("BINARY CLASSIFIER BENCHMARKING WITH MONOTONICITY CONSTRAINTS")
        print("="*80)
        print(f"\nDataset Information:")
        print(f"  Total samples: {len(X)}")
        print(f"  Features: {len(X.columns)}")
        print(f"  Positive class: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
        print(f"  Negative class: {(~y.astype(bool)).sum()} ({(~y.astype(bool)).sum()/len(y)*100:.2f}%)")
        print(f"  Cross-validation folds: {self.n_folds}")
        print(f"  Evaluation metric: Average Precision Score")
        print(f"\nMonotonicity Enforcement:")
        print(f"  - XGBoost/LightGBM: Native monotone_constraints")
        print(f"  - Other models: Post-hoc feature filtering")
        
        for model_name, model in models.items():
            start_time = time.time()
            
            print(f"\n{'='*80}")
            print(f"MODEL: {model_name}")
            print(f"{'='*80}")
            
            # Evaluate with selected features
            results = self.evaluate_model_cv(X, y, model, model_name)
            
            elapsed_time = time.time() - start_time
            results['training_time'] = elapsed_time
            
            self.results.append(results)
            
            print(f"\n  Summary:")
            print(f"    Average Precision: {results['avg_precision_mean']:.4f} ± {results['avg_precision_std']:.4f}")
            print(f"    ROC-AUC: {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
            print(f"    Training time: {elapsed_time:.2f}s")
        
        # Create results dataframe
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values('avg_precision_mean', ascending=False)
        
        return results_df
