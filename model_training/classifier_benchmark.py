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
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, 
    precision_score, recall_score, confusion_matrix,
    precision_recall_curve, roc_curve, compute_sample_weight
)
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
from enum import Enum


class MonotonicityType(Enum):
    """Types of monotonicity checking"""
    MARGINAL = "marginal"      # Check correlation between feature and predictions
    CONDITIONAL = "conditional"  # Check model coefficients (for linear models)
    AUTO = "auto"              # Choose automatically based on model type

class MonotonicityEnforcer:
    """Check and enforce monotonicity constraints on features"""
    
    @staticmethod
    def check_monotonicity(X: pd.DataFrame, y_pred_proba: np.ndarray, 
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
            mask = ~(pd.isna(X[col]) | pd.isna(y_pred_proba))
            if mask.sum() > 0:
                corr, _ = spearmanr(X[col][mask], y_pred_proba[mask])
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
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = 37.02  # Default value, can be overridden
        
        # Generate monotonic constraints for tree-based models
        monotone_constraints_str = self.get_monotone_constraints_string(feature_names)
        
        models = {
            # Tree-based models WITH native monotonic constraints
            'XGBoost': XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                monotone_constraints=monotone_constraints_str,  # NATIVE MONOTONICITY
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'LightGBM': LGBMClassifier(
                scale_pos_weight=scale_pos_weight,
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                monotone_constraints=[1] * len(feature_names),  # NATIVE MONOTONICITY
                random_state=self.random_state,
                verbose=-1
            ),
            'XGBoost (Deep)': XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                max_depth=10,
                learning_rate=0.05,
                n_estimators=200,
                monotone_constraints=monotone_constraints_str,  # NATIVE MONOTONICITY
                random_state=self.random_state,
                eval_metric='logloss'
            ),
            'LightGBM (Deep)': LGBMClassifier(
                scale_pos_weight=scale_pos_weight,
                max_depth=10,
                learning_rate=0.05,
                n_estimators=200,
                monotone_constraints=[1] * len(feature_names),  # NATIVE MONOTONICITY
                random_state=self.random_state,
                verbose=-1
            ),
            
            # Non-tree models (will use post-hoc feature filtering)
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state,
                solver='lbfgs'
            ),
            'Random Forest': RandomForestClassifier(
                class_weight='balanced',
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'SVC': SVC(
                class_weight='balanced',
                kernel='rbf',
                probability=True,
                random_state=self.random_state
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
    
    def verify_monotonicity_posthoc(self, X: pd.DataFrame, y: np.ndarray, 
                                    model, model_name: str,
                                    monotonicity_type: MonotonicityType = MonotonicityType.AUTO) -> Tuple[List[str], Dict]:
        """
        For non-tree models: determine which features to keep based on monotonicity.
        For tree models: verify that native constraints are working.
        
        Args:
            X: Feature matrix
            y: Target vector
            model: Sklearn-compatible model
            model_name: Name of the model
            monotonicity_type: Type of monotonicity to check (MARGINAL, CONDITIONAL, or AUTO)
            
        Returns:
            Tuple of (features_to_use, scores_dict)
        """
        # Use first fold to check monotonicity
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                            random_state=self.random_state)
        
        enforcer = MonotonicityEnforcer()
        
        if monotonicity_type == MonotonicityType.AUTO:
            # For linear models, prefer coefficient check; otherwise use marginal
            if enforcer.supports_coefficient_check(model_name):
                check_type = MonotonicityType.CONDITIONAL
            else:
                check_type = MonotonicityType.MARGINAL
        else:
            check_type = monotonicity_type
        
        if check_type == MonotonicityType.CONDITIONAL:
            if not enforcer.supports_coefficient_check(model_name):
                print(f"  ⚠️  Warning: {model_name} doesn't support coefficient checking. "
                    f"Falling back to marginal monotonicity.")
                check_type = MonotonicityType.MARGINAL
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]

            sample_weight = None
            if model_name == "Gaussian NB":
                sample_weight = compute_sample_weight(
                    class_weight="balanced",
                    y=y_train)
            
            # Scale features for non-tree models
            needs_scaling = model_name not in ['XGBoost', 'LightGBM', 'Random Forest',
                                                'XGBoost (Deep)', 'LightGBM (Deep)']
            if needs_scaling:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                if sample_weight is not None:
                    model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
                else:
                    model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            if self.uses_native_monotonicity(model_name):
                # For tree models with native constraints, just verify using marginal check
                correlations = enforcer.check_monotonicity(X_val, y_pred_proba)
                
                print(f"\n{model_name} - Monotonicity Verification (Native Constraints):")
                print(f"  Using native monotone_constraints - all {len(X.columns)} features enforced")
                
                # Check if any violations (shouldn't happen with native constraints)
                violations = [feat for feat, corr in correlations.items() if corr < 0]
                if violations:
                    print(f"  ⚠️  Warning: {len(violations)} features show negative correlation:")
                    for feat in violations:
                        print(f"    - {feat}: correlation = {correlations[feat]:.4f}")
                else:
                    print(f"  ✓ All features show expected monotonic relationship")
                
                # Return all features (native constraints handle it)
                features_to_use = list(X.columns)
                scores = correlations
                
            else:
                # For non-tree models, check monotonicity
                if check_type == MonotonicityType.CONDITIONAL:
                    # Check coefficients (conditional monotonicity)
                    try:
                        scores = enforcer.check_coefficients(model, list(X.columns))
                        check_name = "Conditional (Coefficient-based)"
                    except (ValueError, AttributeError) as e:
                        print(f"  ⚠️  Could not check coefficients: {e}")
                        print(f"  Falling back to marginal monotonicity check")
                        scores = enforcer.check_monotonicity(X_val, y_pred_proba)
                        check_name = "Marginal (Correlation-based)"
                else:
                    # Check marginal monotonicity
                    scores = enforcer.check_monotonicity(X_val, y_pred_proba)
                    check_name = "Marginal (Correlation-based)"
                
                features_to_use = enforcer.get_monotonic_features(scores, threshold=0.0)
                
                print(f"\n{model_name} - Monotonicity Analysis ({check_name}):")
                print(f"  Original features: {len(X.columns)}")
                print(f"  Monotonic features: {len(features_to_use)}")
                
                non_monotonic = set(X.columns) - set(features_to_use)
                if non_monotonic:
                    print(f"  Excluded (non-monotonic): {len(non_monotonic)} features")
                    for feat in sorted(non_monotonic):
                        score_name = "coefficient" if check_type == MonotonicityType.CONDITIONAL else "correlation"
                        print(f"    - {feat}: {score_name} = {scores[feat]:.4f}")
                
                # Optional: Show top positive contributors
                if features_to_use:
                    top_features = sorted([(feat, scores[feat]) for feat in features_to_use], 
                                        key=lambda x: x[1], reverse=True)[:5]
                    score_name = "coefficient" if check_type == MonotonicityType.CONDITIONAL else "correlation"
                    print(f"  Top 5 monotonic features by {score_name}:")
                    for feat, score in top_features:
                        print(f"    - {feat}: {score:.4f}")
            
            # Only use first fold for analysis
            break
        
        return features_to_use, scores
    
    def evaluate_model_cv(self, X: pd.DataFrame, y: pd.Series, 
                         model, model_name: str, 
                         features_to_use: List[str]) -> Dict:
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
        # Filter to selected features
        X_filtered = X[features_to_use].copy()
        
        # Check if model needs scaling
        needs_scaling = model_name not in ['XGBoost', 'LightGBM', 'Random Forest',
                                            'XGBoost (Deep)', 'LightGBM (Deep)', 'Gaussian NB']
        
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
        
        print(f"\nEvaluating {model_name} with {len(features_to_use)} features...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_filtered, y), 1):
            # Split data
            X_train, X_val = X_filtered.iloc[train_idx], X_filtered.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            sample_weight = None
            if model_name == "Gaussian NB":
                sample_weight = compute_sample_weight(
                    class_weight="balanced",
                    y=y_train)
                
            # Scale if needed
            if needs_scaling:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train and predict
                if sample_weight is not None:
                    model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
                else:
                    model.fit(X_train_scaled, y_train)

                y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                y_pred = model.predict(X_val_scaled)
            else:
                # Train and predict (no scaling)
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                y_pred = model.predict(X_val)
            
            # Calculate metrics
            avg_prec = average_precision_score(y_val, y_pred_proba)
            roc_auc = roc_auc_score(y_val, y_pred_proba)
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            
            # Store metrics
            fold_metrics['avg_precision'].append(avg_prec)
            fold_metrics['roc_auc'].append(roc_auc)
            fold_metrics['f1_score'].append(f1)
            fold_metrics['precision'].append(precision)
            fold_metrics['recall'].append(recall)
            fold_metrics['tn'].append(tn)
            fold_metrics['fp'].append(fp)
            fold_metrics['fn'].append(fn)
            fold_metrics['tp'].append(tp)
            
            print(f"  Fold {fold}: AP={avg_prec:.4f}, ROC-AUC={roc_auc:.4f}, "
                  f"F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        
        # Calculate mean and std across folds
        results = {
            'model': model_name,
            'n_features': len(features_to_use),
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
            
            # Verify monotonicity and get features to use
            features_to_use, correlations = self.verify_monotonicity_posthoc(
                X, y, model, model_name
            )
            # features_to_use, scores = self.verify_monotonicity_posthoc( # This would be to enforce marginal monotonicity
            #     X, y, model, model_name, 
            #     monotonicity_type=MonotonicityType.MARGINAL
            # )
            # features_to_use, scores = self.verify_monotonicity_posthoc( # This would be to enforce conditional monotonicity
            #     X, y, model, model_name,
            #     monotonicity_type=MonotonicityType.CONDITIONAL
            # )
            
            # Store monotonicity information
            self.monotonicity_results.append({
                'model': model_name,
                'uses_native_monotonicity': self.uses_native_monotonicity(model_name),
                'original_features': len(X.columns),
                'features_used': len(features_to_use),
                'excluded_features': list(set(X.columns) - set(features_to_use)),
                'correlations': correlations
            })
            
            # Evaluate with selected features
            results = self.evaluate_model_cv(X, y, model, model_name, features_to_use)
            
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
