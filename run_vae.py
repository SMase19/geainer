import os
import argparse
import numpy as np
import pandas as pd

import models.vae_module as vae

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--algorithm',
                    help='learning rate of the optimizer',
                    default='adage')
parser.add_argument('-l', '--learning_rate',
                    help='learning rate of the optimizer',
                    default=1.1)#0.0005)
parser.add_argument('-b', '--batch_size',
                    help='Number of samples to include in each learning batch',
                    default=50)
parser.add_argument('-e', '--epochs',
                    help='How many times to cycle through the full dataset',
                    default=100)#50)
parser.add_argument('-k', '--kappa',
                    help='How fast to linearly ramp up KL loss',
                    default=1)
parser.add_argument('-d', '--depth',
                    help='Number of layers between input and latent layer',
                    default=1)
parser.add_argument('-c', '--first_layer',
                    help='Dimensionality of the first hidden layer',
                    default=100)
parser.add_argument('-f', '--output_filename',
                    help='The name of the file to store results',
                    default='hyperparam/param.tsv')
parser.add_argument('-g', '--output_boardlog',
                    help='The name of the directory to store tensorboard _logs',
                    default='logs')
parser.add_argument('-s', '--sparsity',
                    help='sparsity',
                    default=0)
parser.add_argument('-n', '--noise',
                    help='noise',
                    default=0.05)
args = parser.parse_args()

# Set hyper parameters
algo = args.algorithm
learning_rate = float(args.learning_rate)
batch_size = int(args.batch_size)
epochs = int(args.epochs)
kappa = float(args.kappa)
depth = int(args.depth)
first_layer = int(args.first_layer)
output_filename = args.output_filename
boardlog_path = args.output_boardlog
noise = args.noise
sparsity = args.sparsity

# Load Gene Expression Data
rnaseq_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv')
rnaseq_df = pd.read_table(rnaseq_file, index_col=0)
print(rnaseq_df.shape)

# Set architecture dimensions
original_dim = rnaseq_df.shape[1]
latent_dim = 100
hidden_dim = 100
epsilon_std = 1.0

if depth == 2:
    latent_dim2 = int(first_layer)

# Random seed
# seed = int(np.random.randint(low=0, high=10000, size=1))
# np.random.seed(seed)
np.random.seed(123)

# Process data
# Split 10% test set randomly
test_set_percent = 0.1
rnaseq_test_df = rnaseq_df.sample(frac=test_set_percent)
rnaseq_train_df = rnaseq_df.drop(rnaseq_test_df.index)
if algo == 'tybalt':
    model = vae.Tybalt(original_dim=original_dim,
                       hidden_dim=hidden_dim,
                       latent_dim=latent_dim,
                       batch_size=batch_size,
                       epochs=epochs,
                       learning_rate=learning_rate,
                       kappa=kappa,
                       epsilon_std=epsilon_std,
                       depth=depth)
elif algo == 'adage':
    model = vae.Adage(original_dim=original_dim,
                      latent_dim=latent_dim,
                      batch_size=batch_size,
                      epochs=epochs,
                      learning_rate=learning_rate,
                      sparsity=sparsity,
                      noise=noise,
                      epsilon_std=epsilon_std)

model.build_encoder_layer()
model.build_decoder_layer()
model.compile_vae()
model.get_summary()
model.train_vae(rnaseq_train_df=rnaseq_train_df, rnaseq_test_df=rnaseq_test_df, boardlog_path=boardlog_path)

model_compressed_df = model.compress(rnaseq_df)
model_compressed_df.columns.name = 'sample_id'
model_compressed_df.columns = model_compressed_df.columns + 1
encoded_file = os.path.join('data', 'encoded_rnaseq_onehidden_warmup_batchnorm.tsv')
model_compressed_df.to_csv(encoded_file, sep='\t')

model_weights = model.get_decoder_weights(rnaseq_df)

encoder_model_file = os.path.join('models', 'encoder_onehidden_vae.hdf5')
decoder_model_file = os.path.join('models', 'decoder_onehidden_vae.hdf5')
model.save_models(encoder_file=encoder_model_file, decoder_file=decoder_model_file)
