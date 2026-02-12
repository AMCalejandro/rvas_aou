#!/usr/bin/env python3
"""
Single model training script for Hail Batch execution.
Trains a specific classifier with monotonicity constraints.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Import the benchmark class (assuming it's in the same directory or package)
from .classifier_benchmark import ClassifierBenchmark, MonotonicityEnforcer, MonotonicityType

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train a single classifier model with monotonicity constraints'
    )

    # Positional arguments
    parser.add_argument(
        'model_name',
        type=str,
        help='Name of the model to train (e.g., XGBoost, LightGBM, "Logistic Regression", etc.)'
    )
    parser.add_argument(
        'input_data',
        type=str,
        help='Path to input CSV/TSV file with features and target'
    )
    parser.add_argument(
        'output_folder',
        type=str,
        help='Path to output folder for results'
    )

    # Optional arguments
    parser.add_argument(
        '--target-column',
        type=str,
        default='target',
        help='Name of the target column (default: target)'
    )

    parser.add_argument(
        '--predictors',
        type=str,
        nargs='+',
        required=False,
        help='List of predictor column names (space separated)'
    )

    parser.add_argument(
        '--predictors-file',
        type=str,
        required=False,
        help='Path to a text file containing predictor names (one per line)'
    )
    parser.add_argument(
        '--framework',
        type=str,
        default='binary',
        help='Framework type (default: binary)'
    )

    threshold_group = parser.add_mutually_exclusive_group()
    
    threshold_group.add_argument(
        '--bin-threshold',
        type=float,
        default=None,
        help='Absolute threshold to binarize the target variable (e.g. 0.7)'
    )

    threshold_group.add_argument(
        '--top-percent',
        type=float,
        default=None,
        help='Use top N%% of values as class 1 (e.g. 5 for top 5%%)'
    )

    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--sep',
        type=str,
        default=',',
        help='Delimiter for input file (default: ,)'
    )

    return parser.parse_args()


class SingleModelTrainer(ClassifierBenchmark):
    """Extended ClassifierBenchmark for single model training"""
    
    def run_single_model(self, X: pd.DataFrame, y: pd.Series, 
                        model_name: str) -> Dict:
        """
        Run training and evaluation for a single model.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_name: Name of the model to train
            
        Returns:
            Dictionary with complete results
        """
        feature_names = list(X.columns)
        models = self.get_models(feature_names)
        
        # Check if model exists
        if model_name not in models:
            available_models = list(models.keys())
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {available_models}"
            )
        
        model = models[model_name]
        
        print("="*80)
        print(f"TRAINING MODEL: {model_name}")
        print("="*80)
        print(f"\nDataset Information:")
        print(f"  Total samples: {len(X)}")
        print(f"  Features: {len(X.columns)}")
        print(f"  Feature names: {list(X.columns)}")
        print(f"  Positive class: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
        print(f"  Negative class: {(~y.astype(bool)).sum()} ({(~y.astype(bool)).sum()/len(y)*100:.2f}%)")
        print(f"  Cross-validation folds: {self.n_folds}")
        print(f"\nMonotonicity Enforcement:")
        if self.uses_native_monotonicity(model_name):
            print(f"  Method: Native monotone_constraints")
        else:
            print(f"  Method: Post-hoc feature filtering")
        
        # Verify monotonicity and get features to use
        print(f"\n{'='*80}")
        print("MONOTONICITY ANALYSIS")
        print(f"{'='*80}")
        
        # features_to_use, scores = self.verify_monotonicity_posthoc(
        #     X, y, model, model_name
        # )
        # features_to_use, scores = self.verify_monotonicity_posthoc( # This would be to enforce marginal monotonicity
        #     X, y, model, model_name, 
        #     monotonicity_type=MonotonicityType.MARGINAL
        # )
        # features_to_use, scores = self.verify_monotonicity_posthoc( # This would be to enforce conditional monotonicity
        #     X, y, model, model_name,
        #     monotonicity_type=MonotonicityType.CONDITIONAL
        # )
        
        # Store monotonicity information
        # monotonicity_info = {
        #     'model': model_name,
        #     'uses_native_monotonicity': self.uses_native_monotonicity(model_name),
        #     'original_features': len(X.columns),
        #     'features_used': len(features_to_use),
        #     'features_list': features_to_use,
        #     'excluded_features': list(set(X.columns) - set(features_to_use)),
        #     'correlations': scores
        # }
        
        # Evaluate model
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION EVALUATION")
        print(f"{'='*80}")
        
        import time
        start_time = time.time()
        
        results = self.evaluate_model_cv(X, y, model, model_name)
        
        elapsed_time = time.time() - start_time
        results['training_time'] = elapsed_time
        
        # Combine results
        complete_results = {
            'model_performance': results,
            'dataset_info': {
                'n_samples': len(X),
                'n_features': len(X.columns),
                'feature_names': list(X.columns),
                'n_positive': int(y.sum()),
                'n_negative': int((~y.astype(bool)).sum()),
                'positive_rate': float(y.sum() / len(y))
            }
        }
        
        # Print summary
        print(f"\n{'='*80}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"\nModel: {model_name}")
        print(f"\nPerformance Metrics:")
        print(f"  Average Precision: {results['avg_precision_mean']:.4f} ± {results['avg_precision_std']:.4f}")
        print(f"  ROC-AUC: {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
        print(f"  F1 Score: {results['f1_score_mean']:.4f} ± {results['f1_score_std']:.4f}")
        print(f"  Precision: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
        print(f"  Recall: {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
        print(f"\nTraining Time: {elapsed_time:.2f}s")
        
        return complete_results

def load_data(
    input_path: str,
    target_column: str,
    predictors: List[str],
    framework: str,
    bin_thresh: Optional[float] = None,
    top_percent: Optional[float] = None,
    sep: str = ','
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load training data from file and prepare features/target.

    Args:
        input_path: Path to input CSV/TSV file
        target_column: Name of target column
        predictors: List of predictor column names
        framework: "continuous" or "binary"
        bin_thresh: Absolute threshold for binarization
        top_percent: Use top N% of values as class 1 (e.g. 5 for top 5%)
        sep: Delimiter
    
    Returns:
        Tuple of (X, y)
    """
    print(f"Loading data from: {input_path}")
    
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith('.tsv'):
        df = pd.read_csv(input_path, sep='\t')
    else:
        df = pd.read_csv(input_path, sep=sep)
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )
    
    if predictors is None:
        predictors = [col for col in df.columns if col != target_column]
    
    missing_predictors = [col for col in predictors if col not in df.columns]
    if missing_predictors:
        raise ValueError(f"Predictors not found in data: {missing_predictors}")
    
    X = df[predictors].copy()
    y = df[target_column].copy()
    
    if framework not in ["continuous", "binary"]:
        raise ValueError("framework must be either 'continuous' or 'binary'")

    unique_vals = set(y.dropna().unique())
    is_binary = unique_vals.issubset({0, 1})

    if framework == "binary":

        if bin_thresh is not None and top_percent is not None:
            raise ValueError("Use either bin_thresh OR top_percent, not both.")

        if not is_binary:
            print("Binary framework selected with continuous target.")

            # --- Absolute threshold ---
            if bin_thresh is not None:
                print(f"Binarizing using absolute threshold = {bin_thresh}")
                y = (y > bin_thresh).astype(int)

            # --- Top N% threshold ---
            elif top_percent is not None:
                if not (0 < top_percent < 100):
                    raise ValueError("top_percent must be between 0 and 100")
                
                percentile_cutoff = np.percentile(y.dropna(), 100 - top_percent)

                print(f"Binarizing using top {top_percent}%")
                print(f"Percentile cutoff value = {percentile_cutoff:.6f}")

                y = (y >= percentile_cutoff).astype(int)

            else:
                raise ValueError(
                    "For binary framework with continuous target, "
                    "provide either bin_thresh or top_percent."
                )
        else:
            print("Target already binary.")
            y = y.astype(int)

    elif framework == "continuous":
        if is_binary:
            print("Continuous framework selected with binary target.")
            print("Keeping binary target as numeric (0/1).")
        y = y.astype(float)

    print(f"Using {len(X.columns)} predictors")
    print(f"Target: {target_column}")
    print("Target distribution:")
    print(y.value_counts(normalize=True))

    return X, y


def load_predictors_from_file(filepath: str) -> List[str]:
    """Load predictor names from text file (one per line)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Predictor file not found: {filepath}")

    with open(filepath, 'r') as f:
        predictors = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]

    if len(predictors) == 0:
        raise ValueError("Predictor file is empty.")

    print(f"Loaded {len(predictors)} predictors from {filepath}")
    return predictors


def save_results(results: Dict, output_folder: str, model_name: str):
    """
    Save results to output folder with standard filenames.
    
    Args:
        results: Results dictionary
        output_folder: Path to output folder
        model_name: Name of the model
    """
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use standard filenames
    json_path = output_path / 'results.json'
    summary_path = output_path / 'summary.txt'
    csv_path = output_path / 'metrics.csv'
    
    # Save JSON results
    with open(json_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        json.dump(convert_types(results), f, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    
    # Save summary text file
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"MODEL TRAINING RESULTS: {model_name}\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-"*80 + "\n")
        dataset_info = results['dataset_info']
        f.write(f"Total samples: {dataset_info['n_samples']}\n")
        f.write(f"Total features: {dataset_info['n_features']}\n")
        f.write(f"Positive class: {dataset_info['n_positive']} ({dataset_info['positive_rate']*100:.2f}%)\n")
        f.write(f"Negative class: {dataset_info['n_negative']} ({(1-dataset_info['positive_rate'])*100:.2f}%)\n\n")
        
        f.write("MONOTONICITY INFORMATION\n")
        f.write("-"*80 + "\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-"*80 + "\n")
        perf = results['model_performance']
        f.write(f"Average Precision: {perf['avg_precision_mean']:.4f} ± {perf['avg_precision_std']:.4f}\n")
        f.write(f"ROC-AUC: {perf['roc_auc_mean']:.4f} ± {perf['roc_auc_std']:.4f}\n")
        f.write(f"F1 Score: {perf['f1_score_mean']:.4f} ± {perf['f1_score_std']:.4f}\n")
        f.write(f"Precision: {perf['precision_mean']:.4f} ± {perf['precision_std']:.4f}\n")
        f.write(f"Recall: {perf['recall_mean']:.4f} ± {perf['recall_std']:.4f}\n")
        f.write(f"\nTraining time: {perf['training_time']:.2f}s\n")
    
    print(f"Summary saved to: {summary_path}")
    
    # Save CSV with key metrics
    perf = results['model_performance']
    metrics_df = pd.DataFrame([{
        'model': model_name,
        'avg_precision_mean': perf['avg_precision_mean'],
        'avg_precision_std': perf['avg_precision_std'],
        'roc_auc_mean': perf['roc_auc_mean'],
        'roc_auc_std': perf['roc_auc_std'],
        'f1_score_mean': perf['f1_score_mean'],
        'f1_score_std': perf['f1_score_std'],
        'precision_mean': perf['precision_mean'],
        'recall_mean': perf['recall_mean'],
        'training_time': perf['training_time'],
    }])
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics CSV saved to: {csv_path}")


def main():
    """Main execution function"""
    args = parse_arguments()
    
    print("="*80)
    print("SINGLE MODEL TRAINING SCRIPT")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Input data: {args.input_data}")
    print(f"  Output folder: {args.output_folder}")
    print(f"  Target column: {args.target_column}")
    print(f" The threshold for binarization {args.bin_threshold}")
    print(f"  CV folds: {args.n_folds}")
    print(f"  Random state: {args.random_state}")
    print()
    
    try:
        # Load data
        predictors = (
            load_predictors_from_file(args.predictors_file)
            if args.predictors_file
            else args.predictors
        )
        X, y = load_data(args.input_data, args.target_column, predictors, args.framework, args.bin_threshold, args.top_percent, args.sep)

        # Initialize trainer
        trainer = SingleModelTrainer(
            n_folds=args.n_folds,
            random_state=args.random_state
        )
        
        # Run training for single model
        results = trainer.run_single_model(X, y, args.model_name)
        
        # Save results
        save_results(results, args.output_folder, args.model_name)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
