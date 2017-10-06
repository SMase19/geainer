import os
import argparse
import subprocess
import pandas as pd


def get_param(param):
    try:
        sweep = parameter_df.loc[param, 'sweep']
    except:
        sweep = ''
        print("Warning! No parameter detected of name: {}".format(param))
    return sweep.split(',')


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--parameter_file',
                    help='location of tab separated parameter file to sweep')
parser.add_argument('-c', '--config_file',
                    help='location of the configuration file for PMACS')
parser.add_argument('-a', '--algorithm',
                    help='which algorithm to sweep hyperparameters over')
parser.add_argument('-s', '--python_path',
                    help='absolute path of python version',
                    default='~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python')
parser.add_argument('-d', '--param_folder',
                    help='folder to store param sweep results',
                    default='sweep_param')
parser.add_argument('-t', '--script',
                    help='path the script to run the parameter sweep over',
                    default='keras_vae.py')
args = parser.parse_args()

parameter_file = args.parameter_file
config_file = args.config_file
algorithm = args.algorithm
python_path = args.python_path
param_folder = args.param_folder
script = args.script

if not os.path.exists(param_folder):
    os.makedirs(param_folder)

# Load data
parameter_df = pd.read_table(parameter_file, index_col=0)
config_df = pd.read_table(config_file, index_col=0)

# Retrieve hyperparameters to sweep over
learning_rates = get_param('learning_rate')
batch_sizes = get_param('batch_size')
epochs = get_param('epochs')
kappas = get_param('kappa')
sparsities = get_param('sparsity')
noises = get_param('noise')
depth = get_param('depth')
first_layer_dim = get_param('first_layer_dim')

# Build lists of job commands depending on input algorithm
all_commands = []
if algorithm == 'tybalt':
    for lr in learning_rates:
        for bs in batch_sizes:
            for e in epochs:
                for k in kappas:
                    f = 'sweepparam_{}lr_{}bs_{}e_{}k.tsv'.format(lr, bs, e, k)
                    f = os.path.join(param_folder, f)
                    params = ['--learning_rate', lr,
                              '--batch_size', bs,
                              '--epochs', e,
                              '--kappa', k,
                              '--output_filename', f,
                              '--depth', depth[0],
                              '--first_layer', first_layer_dim[0]]
                    final_command = [python_path, script] + params
                    all_commands.append(final_command)
elif algorithm == 'adage':
    for lr in learning_rates:
        for bs in batch_sizes:
            for e in epochs:
                for s in sparsities:
                    for n in noises:
                        f = 'sweepparam_{}lr_{}bs_{}e_{}s_{}n.tsv'.format(lr, bs, e, s, n)
                        f = os.path.join(param_folder, f)
                        params = ['--learning_rate', lr,
                                  '--batch_size', bs,
                                  '--epochs', e,
                                  '--sparsity', s,
                                  '--noise', n,
                                  '--output_filename', f]
                        final_command = [python_path, script] + params
                        all_commands.append(final_command)

for command in all_commands:
    subprocess.call(command)
