#!/usr/bin/env python3
"""
Helper script to list available models for training.
"""

from ..classifier_benchmark import ClassifierBenchmark

def main():
    """Print list of available models"""
    print("="*80)
    print("AVAILABLE MODELS FOR TRAINING")
    print("="*80)
    print()
    
    # Create temporary instance to get model list
    benchmark = ClassifierBenchmark()
    models = benchmark.get_models(feature_names=['dummy'])  # Feature names don't matter for listing
    
    print("You can train any of the following models:")
    print()
    
    # Group by monotonicity support
    native_mono = []
    posthoc_mono = []
    
    for model_name in models.keys():
        if benchmark.uses_native_monotonicity(model_name):
            native_mono.append(model_name)
        else:
            posthoc_mono.append(model_name)
    
    print("Models with NATIVE monotonicity constraints:")
    print("  (Monotonicity enforced directly in training)")
    for i, model in enumerate(native_mono, 1):
        print(f"  {i}. {model}")
    
    print()
    print("Models with POST-HOC monotonicity filtering:")
    print("  (Features filtered based on monotonicity)")
    for i, model in enumerate(posthoc_mono, 1):
        print(f"  {i}. {model}")
    
    print()
    print("="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    print()
    print("Single model:")
    print('  python hail_batch_runner.py "XGBoost" data.csv output/ --target-column label')
    print()
    print("Multiple models (comma-separated):")
    print('  python hail_batch_runner.py "XGBoost,LightGBM,Logistic Regression" data.csv output/')
    print()
    print("All models:")
    all_models = ','.join(models.keys())
    print(f'  python hail_batch_runner.py "{all_models}" data.csv output/')
    print()
    
    print("Note: Model names with spaces must be quoted!")
    print()


if __name__ == "__main__":
    main()