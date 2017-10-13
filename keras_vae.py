'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import os
import argparse
import numpy as np
import pandas as pd

from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras import backend as K
from keras import metrics, optimizers
from keras.callbacks import Callback, TensorBoard


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--learning_rate',
                    help='learning rate of the optimizer',
                    default=0.0005)
parser.add_argument('-b', '--batch_size',
                    help='Number of samples to include in each learning batch',
                    default=50)
parser.add_argument('-e', '--epochs',
                    help='How many times to cycle through the full dataset',
                    default=50)
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
                    help='The name of the directory to store tensorboard _logs')
args = parser.parse_args()


# Set hyper parameters
learning_rate = float(args.learning_rate)
batch_size = int(args.batch_size)
epochs = int(args.epochs)
kappa = float(args.kappa)
depth = int(args.depth)
first_layer = int(args.first_layer)
output_filename = args.output_filename
boardlog_path = args.output_boardlog

# Load Gene Expression Data
rnaseq_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv')
rnaseq_df = pd.read_table(rnaseq_file, index_col=0)
print(rnaseq_df.shape)

# Set architecture dimensions
original_dim = rnaseq_df.shape[1]
latent_dim = 100
epsilon_std = 1.0
beta = K.variable(0)
if depth == 2:
    latent_dim2 = int(first_layer)

# Random seed
seed = int(np.random.randint(low=0, high=10000, size=1))
np.random.seed(seed)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    # def binary_crossentropy(self, y_true, y_pred):
    #     return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)
    #
    # def vae_loss(self, x, x_decoded_mean):
    #     xent_loss = K.mean(self.binary_crossentropy(x, x_decoded_mean),axis=-1)
    #     kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) -
    #                             K.exp(z_log_var_encoded), axis=-1)
    #     return K.mean(xent_loss + (K.get_value(beta) * kl_loss))

    # wrong?
    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var_encoded - K.square(z_mean_encoded) -
                                K.exp(z_log_var_encoded), axis=-1)
        return K.mean(xent_loss + (K.get_value(beta) * kl_loss))

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x


class WarmUpCallback(Callback):
    def __init__(self, beta, kappa):
        self.beta = beta
        self.kappa = kappa

    # Behavior on each epoch
    def on_epoch_end(self, epoch, logs={}):
        if K.get_value(self.beta) <= 1:
            K.set_value(self.beta, K.get_value(self.beta) + self.kappa)

# Process data
# Split 10% test set randomly
test_set_percent = 0.1
rnaseq_test_df = rnaseq_df.sample(frac=test_set_percent)
rnaseq_train_df = rnaseq_df.drop(rnaseq_test_df.index)

# Input place holder for RNAseq data with specific input size
rnaseq_input = Input(shape=(original_dim, ))

# ~~~~~~~~~~~~~~~~~~~~~~
# ENCODER
# ~~~~~~~~~~~~~~~~~~~~~~
if depth == 1:
    z_shape = latent_dim
    z_mean_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(rnaseq_input)
    z_log_var_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(rnaseq_input)

elif depth == 2:
    z_shape = latent_dim2
    hidden_dense_linear = Dense(latent_dim, kernel_initializer='glorot_uniform')(rnaseq_input)
    hidden_dense_batchnorm = BatchNormalization()(hidden_dense_linear)
    hidden_encoded = Activation('relu')(hidden_dense_batchnorm)

    z_mean_dense_linear = Dense(latent_dim2, kernel_initializer='glorot_uniform')(hidden_encoded)
    z_log_var_dense_linear = Dense(latent_dim2, kernel_initializer='glorot_uniform')(hidden_encoded)

z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)

z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)

z = Lambda(sampling,
           output_shape=(z_shape, ))([z_mean_encoded, z_log_var_encoded])

# ~~~~~~~~~~~~~~~~~~~~~~
# DECODER
# ~~~~~~~~~~~~~~~~~~~~~~
if depth == 1:
    decoder_to_reconstruct = Dense(original_dim,
                                   kernel_initializer='glorot_uniform',
                                   activation='sigmoid')
elif depth == 2:
    decoder_to_reconstruct = Sequential()
    decoder_to_reconstruct.add(Dense(latent_dim,
                                     kernel_initializer='glorot_uniform',
                                     activation='relu',
                                     input_dim=latent_dim2))
    decoder_to_reconstruct.add(Dense(original_dim,
                                     kernel_initializer='glorot_uniform',
                                     activation='sigmoid'))

rnaseq_reconstruct = decoder_to_reconstruct(z)

# ~~~~~~~~~~~~~~~~~~~~~~
# CONNECTIONS
# ~~~~~~~~~~~~~~~~~~~~~~
adam = optimizers.Adam(lr=learning_rate)
vae_layer = CustomVariationalLayer()([rnaseq_input, rnaseq_reconstruct])
vae = Model(rnaseq_input, vae_layer)
vae.compile(optimizer=adam, loss=None, loss_weights=[beta])
vae.summary()

# fit Model
hist = vae.fit(np.array(rnaseq_train_df),
               shuffle=True,
               epochs=epochs,
               batch_size=batch_size,
               validation_data=(np.array(rnaseq_test_df),
                                np.array(rnaseq_test_df)),
               callbacks=[WarmUpCallback(beta, kappa),
                          TensorBoard(log_dir=boardlog_path)])

encoder = Model(rnaseq_input, z_mean_encoded)
encoded_rnaseq_df = encoder.predict_on_batch(rnaseq_df)
encoded_rnaseq_df = pd.DataFrame(encoded_rnaseq_df, index=rnaseq_df.index)

encoded_rnaseq_df.columns.name = 'sample_id'
encoded_rnaseq_df.columns = encoded_rnaseq_df.columns + 1
encoded_file = os.path.join('data', 'encoded_rnaseq_onehidden_warmup_batchnorm.tsv')
encoded_rnaseq_df.to_csv(encoded_file, sep='\t')

decoder_input = Input(shape=(latent_dim,))  # can generate from any sampled z vector
_x_decoded_mean = decoder_to_reconstruct(decoder_input)
decoder = Model(decoder_input, _x_decoded_mean)

encoder_model_file = os.path.join('models', 'encoder_onehidden_vae.hdf5')
decoder_model_file = os.path.join('models', 'decoder_onehidden_vae.hdf5')

encoder.save(encoder_model_file)
decoder.save(decoder_model_file)

# Save training performance
history_df = pd.DataFrame(hist.history)
history_df = history_df.assign(learning_rate=learning_rate)
history_df = history_df.assign(batch_size=batch_size)
history_df = history_df.assign(epochs=epochs)
history_df = history_df.assign(kappa=kappa)
history_df = history_df.assign(seed=seed)
history_df = history_df.assign(depth=depth)
history_df = history_df.assign(first_layer=first_layer)
history_df.to_csv(output_filename, sep='\t')