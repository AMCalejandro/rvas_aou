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
    '--predictors-file',
    type=str,
    default=None,
    help='Path to file containing predictor column names (optional)'
)
parser.add_argument(
    '--framework',
    type=str,
    default='binary',
    help='Framework type (default: binary)'
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
    name=f"Model training: {', '.join(model_types)}"
)

training_data = b.read_input(args.input_data)

predictors_file = None
if args.predictors_file:
    predictors_file = b.read_input(args.predictors_file)

for model in model_types:
    j = b.new_job(name=f'Train-{model.replace(" ", "-")}')
    
    j._machine_type = config['model-training']['machine-type']
    j.storage('10Gi')  # Sufficient for pixi, model, and outputs
    
    j.command('apt update')
    j.command('apt install -y git curl moreutils')
    
    j.command('git clone -b scallion-interpretation https://github.com/AMCalejandro/rvas_aou.git')
    j.command('cd rvas_aou')
    
    j.command('curl -fsSL https://pixi.sh/install.sh | sh')
    j.command('export PATH=/root/.pixi/bin:$PATH')
    j.command('pixi install')
    
    # Build the training command
    training_cmd = (
        f'pixi run python -m model_training.run_model '
        f'{quote(model)} '
        f'{quote(training_data)} '
        f'./output/ '
        f'--target-column {quote(args.target_column)} '
    )
    
    if predictors_file:
        training_cmd += f'--predictors-file {quote(predictors_file)} '
    
    training_cmd += (
        f'--framework {quote(args.framework)} '
        f'--n-folds {args.n_folds} '
        f'--random-state {args.random_state}'
    )
    
    j.command(training_cmd)
    
    # Move standard output files to job output files
    j.command(f'mv ./output/results.json {j.ofile1}')
    j.command(f'mv ./output/summary.txt {j.ofile2}')
    j.command(f'mv ./output/metrics.csv {j.ofile3}')
    
    model_name = model.replace(" ", "-")
    b.write_output(j.ofile1, f'{output_folder}/{args.target_column}/{model_name}_results.json')
    b.write_output(j.ofile2, f'{output_folder}/{model_name}_summary.txt')
    b.write_output(j.ofile3, f'{output_folder}/{model_name}_metrics.csv')
    print(f"Added job for model: {model}")

print(f"\nSubmitting batch with {len(model_types)} model training jobs...")
b.run()
print("Batch submitted successfully!")