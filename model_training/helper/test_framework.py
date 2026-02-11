#!/usr/bin/env python3
"""
Test script to verify the model training framework is working correctly.
Creates synthetic data and runs a quick test.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

def create_synthetic_data(n_samples=1000, n_features=5, random_state=42):
    """
    Create synthetic binary classification dataset with monotonic features.
    
    All features are positively correlated with the target to ensure
    monotonicity constraints are satisfied.
    """
    np.random.seed(random_state)
    
    # Create features from normal distribution
    X = np.random.randn(n_samples, n_features)
    
    # Make features positive (monotonic increasing)
    X = np.abs(X)
    
    # Create target based on weighted sum of features + noise
    weights = np.random.rand(n_features)
    linear_combination = X @ weights
    
    # Add noise
    noise = np.random.randn(n_samples) * 0.5
    score = linear_combination + noise
    
    # Convert to binary target (imbalanced)
    threshold = np.percentile(score, 70)  # ~30% positive class
    y = (score > threshold).astype(int)
    
    # Create dataframe
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df


def test_single_model_training():
    """Test single model training"""
    print("="*80)
    print("TEST 1: Single Model Training")
    print("="*80)
    
    # Create synthetic data
    print("\n1. Creating synthetic dataset...")
    df = create_synthetic_data(n_samples=500, n_features=3)
    
    # Save to temporary file
    test_data_path = Path('./test_data.csv')
    df.to_csv(test_data_path, index=False)
    print(f"   Created {len(df)} samples with {len(df.columns)-1} features")
    print(f"   Positive class: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
    
    # Test single model training
    print("\n2. Testing single model training...")
    try:
        from ..run_model import SingleModelTrainer
        
        X = df.drop('target', axis=1)
        y = df['target']
        
        trainer = SingleModelTrainer(n_folds=3, random_state=42)
        results = trainer.run_single_model(X, y, "XGBoost")
        
        print("\n   ✓ Single model training successful!")
        print(f"   Average Precision: {results['model_performance']['avg_precision_mean']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n   ✗ Single model training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model_listing():
    """Test model listing"""
    print("\n" + "="*80)
    print("TEST 2: Model Listing")
    print("="*80)
    
    try:
        from ..classifier_benchmark import ClassifierBenchmark
        
        benchmark = ClassifierBenchmark()
        models = benchmark.get_models(['feature1', 'feature2'])
        
        print(f"\n   Found {len(models)} available models:")
        for i, model_name in enumerate(models.keys(), 1):
            native = "✓" if benchmark.uses_native_monotonicity(model_name) else " "
            print(f"   [{native}] {i}. {model_name}")
        
        print("\n   ✓ Model listing successful!")
        return True
        
    except Exception as e:
        print(f"\n   ✗ Model listing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test data loading function"""
    print("\n" + "="*80)
    print("TEST 3: Data Loading")
    print("="*80)
    
    try:
        from ..run_model import load_data
        
        # Create and save test data
        df = create_synthetic_data(n_samples=100, n_features=3)
        test_file = Path('./test_load.csv')
        df.to_csv(test_file, index=False)
        
        # Test loading
        X, y = load_data(str(test_file), 'target')
        
        print(f"\n   Loaded {len(X)} samples with {len(X.columns)} features")
        print(f"   Target has {y.sum()} positive samples")
        
        # Cleanup
        test_file.unlink()
        
        print("\n   ✓ Data loading successful!")
        return True
        
    except Exception as e:
        print(f"\n   ✗ Data loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("MODEL TRAINING FRAMEWORK - TEST SUITE")
    print("="*80)
    
    results = []
    
    # Run tests
    results.append(("Model Listing", test_model_listing()))
    results.append(("Data Loading", test_data_loading()))
    results.append(("Single Model Training", test_single_model_training()))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status}: {test_name}")
    
    # Overall result
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nThe framework is ready to use!")
        print("\nNext steps:")
        print("  1. Edit config.yaml with your Hail Batch settings")
        print("  2. Prepare your training data")
        print("  3. Run: python hail_batch_runner.py 'XGBoost' data.csv output/")
        return 0
    else:
        print("SOME TESTS FAILED! ✗")
        print("="*80)
        print("\nPlease fix the errors above before using the framework.")
        return 1


if __name__ == "__main__":
    sys.exit(main())