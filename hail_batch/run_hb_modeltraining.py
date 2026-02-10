import hailtop.batch as hb
from shlex import quote
import yaml
import argparse
from urllib.parse import urlparse

parser = argparse.ArgumentParser(description='run scallion model training on Hail Batch', prefix_chars='@')
parser.add_argument('models', type=str, help='comma separated list of models to run')
parser.add_argument('input_data', type=str, help='This is the data for training')
parser.add_argument('output_folder', type=str, help='output folder')
args = parser.parse_args()

# print(args.input_data)
# input_path = urlparse(args.input_data)
# print(input_path)
# bucket_name = plate_path.netloc
# input_folder = plate_path.path.rstrip('/')
output_folder = args.output_folder.rstrip('/')
model_types = [model.strip() for model in args.models.split(',')]
# gs://aou_amc/scallion/training/data/scallion_vsms.tsv
# gs://aou_amc/scallion/training/output/

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

backend = hb.ServiceBackend(billing_project=config['hail-batch']['billing-project'],
                            remote_tmpdir=config['hail-batch']['remote-tmpdir'],
                            regions=config['hail-batch']['regions'])

b = hb.Batch(backend=backend, name=f"Run model training with {args.models}")


training_eval_data = b.read_input(args.input_data)




for model in model_types:
    j = b.new_job(name=f'Model training {model} ')
    # j.cloudfuse(bucket_name, '/images')
    j._machine_type = config['model-training']['machine-type']
    j.storage('20Gi') # should be large enough for pixi (12 GB), model and tsv output (not for images)

    num_workers = config['model-training']['num-workers']

    j.command('apt update')
    j.command('apt install -y git curl moreutils')
    j.command('git clone ')
    j.command('cd microscopy_computational_tools')
    j.command('curl -fsSL https://pixi.sh/install.sh | sh')
    j.command('export PATH=/root/.pixi/bin:$PATH')
    j.command('pixi install')


#     j.command(f'pixi run python embeddings/run_model.py {args.model} {model_weights} /images/{quote(image_folder)} {quote(args.channel_names)} {quote(args.channel_substrings)} {quote(centers_file)} {num_workers} embedding.tsv crops.png')
#     j.command(f'mv embedding.tsv {j.ofile1}')
#     j.command(f'mv crops.png {j.ofile2}')
#     b.write_output(j.ofile1, f'{output_folder}/embedding_{args.model}_{plate}.tsv')
#     b.write_output(j.ofile2, f'{output_folder}/embedding_{args.model}_{plate}.png')
b.run()