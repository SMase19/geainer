import os
import argparse

import itertools
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
parser.add_argument('-a', '--algorithm',
                    help='which algorithm to sweep hyperparameters over',
                    default='tybalt')
parser.add_argument('-p', '--python_path',
                    help='absolute path of python version',
                    default='~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python')
parser.add_argument('-s', '--script',
                    help='path the script to run the parameter sweep over',
                    default='run_vae.py')
parser.add_argument('-t', '--cancer_type',
                    help='path the script to run the parameter sweep over',
                    default='pancan')
args = parser.parse_args()

algorithm = args.algorithm
parameter_file = 'config/' + algorithm + '_param_sweep.tsv'
python_path = args.python_path
script = args.script
c_type = args.cancer_type

# Load data
parameter_df = pd.read_table(parameter_file, index_col=0)

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
if algorithm == 'tybalt':
    for lr, bs, e, k in itertools.product(learning_rates, batch_sizes, epochs, kappas):
        board_log_path = 'data/' + c_type + '/' + algorithm + \
                         'logs/learning={},batch={},epochs={},kappa={},depth={},first={}'.format(
                            lr, bs, e, k, depth[0], first_layer_dim[0])
        final_command = [python_path, script,
                         '--algorithm', algorithm,
                         '--learning_rate', lr,
                         '--batch_size', bs,
                         '--epochs', e,
                         '--kappa', k,
                         '--depth', depth[0],
                         '--first_layer', first_layer_dim[0],
                         '--output_board_log', board_log_path]
        try:
            exe_command = " ".join(final_command)
        except:
            exe_command = final_command
        subprocess.call(exe_command, shell=True)
        os.system("bash confirm_features.sh")
        os.system("mv figures/tsne_vae.pdf " + board_log_path)
        os.system("mv figures/tsne_vae.png " + board_log_path)
        os.system("mv models/decoder_vae.hdf5 " + board_log_path)
        os.system("mv models/encoder_vae.hdf5 " + board_log_path)
        os.system("mv results/tsne_features.tsv " + board_log_path)
        os.system("mv data/encoded_rnaseq.tsv " + board_log_path)
    # for lr in learning_rates:
    #     for bs in batch_sizes:
    #         for e in epochs:
    #             for k in kappas:
    #                 board_log_path = 'logs/learning={},batch={},epochs={},kappa={},depth={},first={}'.format(
    #                     lr, bs, e, k, depth[0], first_layer_dim[0])
    #                 final_command = [python_path, script,
    #                                  '--learning_rate', lr,
    #                                  '--batch_size', bs,
    #                                  '--epochs', e,
    #                                  '--kappa', k,
    #                                  '--depth', depth[0],
    #                                  '--first_layer', first_layer_dim[0],
    #                                  '--output_board_log', board_log_path]
    #                 try:
    #                     exe_command = " ".join(final_command)
    #                 except:
    #                     exe_command = final_command
    #                 subprocess.call(exe_command, shell=True)
    #                 os.system("bash confirm_features.sh")
    #                 os.system("mv figures/tsne_vae.pdf " + board_log_path)
    #                 os.system("mv figures/tsne_vae.png " + board_log_path)
    #                 os.system("mv models/decoder_onehidden_vae.hdf5 " + board_log_path)
    #                 os.system("mv models/encoder_onehidden_vae.hdf5 " + board_log_path)

elif algorithm == 'adage':
    for lr, bs, e, s, n in itertools.product(learning_rates, batch_sizes, epochs, sparsities, noises):
        board_log_path = 'data/' + c_type + '/' + algorithm + \
                         'logs/learning={},batch={},epochs={},sparsity={},noise={}'.format(lr, bs, e, s, n)
        final_command = [python_path, script,
                         '--learning_rate', lr,
                         '--batch_size', bs,
                         '--epochs', e,
                         '--sparsity', s,
                         '--noise', n,
                         '--output_board_log', board_log_path]
        try:
            exe_command = " ".join(final_command)
        except:
            exe_command = final_command
        subprocess.call(exe_command, shell=True)
        os.system("bash confirm_features.sh")
        os.system("mv figures/tsne_vae.pdf " + board_log_path)
        os.system("mv figures/tsne_vae.png " + board_log_path)
        os.system("mv models/decoder_vae.hdf5 " + board_log_path)
        os.system("mv models/encoder_vae.hdf5 " + board_log_path)
        os.system("mv results/tsne_features.tsv " + board_log_path)
        os.system("mv data/encoded_rnaseq.tsv " + board_log_path)



    # for lr in learning_rates:
    #     for bs in batch_sizes:
    #         for e in epochs:
    #             for s in sparsities:
    #                 for n in noises:
    #                     boardlog_path = 'logs/learning={},batch={},epochs={},sparsity={},noise={}'.format(
    #                         lr, bs, e, s, n)
    #                     final_command = [python_path, script,
    #                                      '--learning_rate', lr,
    #                                      '--batch_size', bs,
    #                                      '--epochs', e,
    #                                      '--sparsity', s,
    #                                      '--noise', n,
    #                                      '--output_board_log', boardlog_path]
    #                     try:
    #                         exe_command = " ".join(final_command)
    #                     except:
    #                         exe_command = final_command
    #                     subprocess.call(exe_command, shell=True)
    #                     os.system("bash confirm_features.sh")
    #                     os.system("mv figures/tsne_vae.pdf " + boardlog_path)
    #                     os.system("mv figures/tsne_vae.png " + boardlog_path)
    #                     os.system("mv models/decoder_onehidden_vae.hdf5 " + boardlog_path)
    #                     os.system("mv models/encoder_onehidden_vae.hdf5 " + boardlog_path)
    #                     os.system("mv results/tybalt_tsne_features.tsv " + boardlog_path)
