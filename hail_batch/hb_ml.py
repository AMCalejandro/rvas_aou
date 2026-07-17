#!/usr/bin/env python3

import hailtop.batch as hb
from shlex import quote
import yaml
import argparse
from urllib.parse import urlparse

parser = argparse.ArgumentParser(
    description='Run scallion model training on Hail Batch'
)
parser.add_argument(
    '--models', 
    type=str, 
    help='Comma separated list of models to run (e.g., "XGBoost,LightGBM,Logistic Regression")'
)
parser.add_argument(
    '--input_data', 
    type=str, 
    help='Path to input training data (CSV/TSV file)'
)
parser.add_argument(
    '--output_folder', 
    type=str, 
    help='Output folder for results'
)
parser.add_argument(
    '--target-column',
    type=str,
    default='target',
    help='Name of target column in dataset (default: target)'
)
parser.add_argument(
    '--predictors',
    type=str,
    default=None,
    help='Comma-separated list of predictor column names (optional)'
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
    '--tune',
    action='store_true',
    help='If set, run hyperparameter tuning with Optuna instead of standard training'
)

args = parser.parse_args()

output_folder = args.output_folder.rstrip('/')
model_types = [model.strip() for model in args.models.split(',')]

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

backend = hb.ServiceBackend(
    billing_project=config['hail-batch']['billing-project'],
    remote_tmpdir=config['hail-batch']['remote-tmpdir'],
    regions=config['hail-batch']['regions']
)

b = hb.Batch(
    backend=backend,
    default_image=config['hail-batch']['docker-image'],
    name=f"Model training: {', '.join(model_types)}"
)

training_data = b.read_input(args.input_data)

def model_requires_scaler(model_name: str) -> bool:
    no_scaler_models = {'XGBoost', 'LightGBM', 'Random Forest',
                        'XGBoost (Deep)', 'LightGBM (Deep)',}
    return model_name not in no_scaler_models

# predictors_file = None
# if args.predictors_file:
#     predictors_file = b.read_input(args.predictors_file)

for model in model_types:
    j = b.new_job(name=f'Train-{model.replace(" ", "-")}')
    
    if not args.tune:
        j._machine_type = config['model-training']['machine-type']
        j.storage('10Gi')  # Sufficient for pixi, model, and outputs
    else:
        j._machine_type = config['model-tuning']['machine-type']
        
    
    j.command('git clone https://github.com/AMCalejandro/rvas_aou.git')
    j.command('cd rvas_aou')
    j.command('pixi install')
    
    # Build the training command
    if not args.tune:
        training_cmd = (
            f'pixi run python -m model_training.run_model '
            f'{quote(model)} '
            f'{quote(training_data)} '
            f'./output/ '
            f'--target-column {quote(args.target_column)} '
        )
        if args.predictors:
            training_cmd += f'--predictors {quote(args.predictors)} '
        
        training_cmd += (
            f'--framework {quote(args.framework)} '
            f'--n-folds {args.n_folds} '
            f'--random-state {args.random_state} '
        )

        if args.bin_threshold is not None:
            training_cmd += f'--bin-threshold {args.bin_threshold} '
        if args.top_percent is not None:
            training_cmd += f'--top-percent {args.top_percent} '
        
        j.command(training_cmd)
    
    else:
        tune_cmd = (
            f'pixi run python -m model_training.tune_hyperparam.tune_model '
            f'{quote(model)} '
            f'{quote(training_data)} '
            f'./output/ '
            f'--target-column {quote(args.target_column)} '
        )

        if args.predictors:
            training_cmd += f'--predictors {quote(args.predictors)} '

        tune_cmd += (
            f'--framework {quote(args.framework)} '
            f'--n-folds {args.n_folds} '
            f'--random-state {args.random_state} '
        )

        if args.bin_threshold is not None:
            tune_cmd += f'--bin-threshold {args.bin_threshold} '
        if args.top_percent is not None:
            tune_cmd += f'--top-percent {args.top_percent} '

        j.command(tune_cmd)
    
    # Move standard output files to job output files
    j.command(f'mv ./output/results.json {j.ofile1}')
    j.command(f'mv ./output/summary.txt {j.ofile2}')
    j.command(f'mv ./output/metrics.csv {j.ofile3}')
    
    j.command(f'mv ./output/model.pkl {j.ofile4}')
    j.command(f'mv ./output/model_metadata.json {j.ofile5}')
    
    model_name = model.replace(" ", "-")
    b.write_output(j.ofile1, f'{output_folder}/{args.target_column}/{model_name}_results.json')
    b.write_output(j.ofile2, f'{output_folder}/{args.target_column}/{model_name}_summary.txt')
    b.write_output(j.ofile3, f'{output_folder}/{args.target_column}/{model_name}_metrics.csv')
    b.write_output(j.ofile4, f'{output_folder}/{args.target_column}/models/{model_name}.pkl')
    b.write_output(j.ofile5, f'{output_folder}/{args.target_column}/models/{model_name}_metadata.json')
    
    if model_requires_scaler(model):
        j.command(f'mv ./output/scaler.pkl {j.ofile6}')
        b.write_output(j.ofile6, f'{output_folder}/{args.target_column}/models/{model_name}_scaler.pkl')

    if args.framework == 'binary':
        j.command(f'mv ./output/pr_curve_folds.json {j.ofile7}')
        j.command(f'mv ./output/pr_curve_mean.json  {j.ofile8}')
        b.write_output(j.ofile7, f'{output_folder}/{args.target_column}{model_name}_pr_curve_folds.json')
        b.write_output(j.ofile8, f'{output_folder}/{args.target_column}{model_name}_pr_curve_mean.json')
    elif args.framework == 'regression':
        j.command(f'mv ./output/scatter_folds.json        {j.ofile7}')
        j.command(f'mv ./output/actual_vs_predicted.json  {j.ofile8}')
        b.write_output(j.ofile7, f'{output_folder}/{args.target_column}{model_name}_scatter_folds.json')
        b.write_output(j.ofile8, f'{output_folder}/{args.target_column}{model_name}_actual_vs_predicted.json')


print(f"\nSubmitting batch with {len(model_types)} model training jobs...")
b.run()
print("Batch submitted successfully!")