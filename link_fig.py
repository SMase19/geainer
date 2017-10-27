# -*- coding: utf-8 -*-
from PyPDF2 import PdfFileMerger
import argparse
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
parser.add_argument('-p', '--parameter_file',
                    help='location of tab separated parameter file to sweep',
                    default='config/tybalt_param_sweep.tsv')
args = parser.parse_args()

parameter_file = args.parameter_file
algorithm = args.algorithm
parameter_df = pd.read_table(parameter_file, index_col=0)
# config_file = args.config_file
learning_rates = get_param('learning_rate')
batch_sizes = get_param('batch_size')
epochs = get_param('epochs')
kappas = get_param('kappa')
sparsities = get_param('sparsity')
noises = get_param('noise')
depth = get_param('depth')
first_layer_dim = get_param('first_layer_dim')

merger = PdfFileMerger()
if algorithm == 'tybalt':
    for lr in learning_rates:
        for bs in batch_sizes:
            for e in epochs:
                for k in kappas:
                    param_folder = 'learning={},batch={},epochs={},kappa={},depth={},first={}'.format(
                        lr, bs, e, k, depth[0], first_layer_dim[0])
                    merger.append('_logs/'+param_folder+'/tsne_vae.pdf', bookmark=param_folder)

merger.write('merged.pdf')
merger.close()
