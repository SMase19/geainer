import numpy as np
import pandas as pd
import copy
import os

from keras.layers import Input, Dense, Lambda, Layer, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1
from keras.models import Model, Sequential
from keras import backend
from keras import metrics, optimizers
from keras.callbacks import Callback, TensorBoard


class CustomVariationalLayer(Layer):
    def __init__(self, original_dim, z_log_var_encoded, z_mean_encoded, beta, **kwargs):
        self.is_placeholder = True
        self.original_dim = original_dim
        self.z_log_var_encoded = z_log_var_encoded
        self.z_mean_encoded = z_mean_encoded
        self.beta = beta
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
        xent_loss = self.original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * backend.sum(1 + self.z_log_var_encoded - backend.square(self.z_mean_encoded) -
                                backend.exp(self.z_log_var_encoded), axis=-1)
        return backend.mean(xent_loss + (backend.get_value(self.beta) * kl_loss))

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
        if backend.get_value(self.beta) <= 1:
            backend.set_value(self.beta, backend.get_value(self.beta) + self.kappa)


class AutoEncoderBase():

    def build_encoder_layer(self):
        pass

    def build_decoder_layer(self):
        pass

    def compile_vae(self):
        pass

    def get_summary(self):
        pass

    def train_vae(self, rnaseq_train_df, rnaseq_test_df, boardlog_path):
        pass

    def visualize_training(self, output_file):
        pass

    def compress(self, df):
        pass

    def get_decoder_weights(self, df):
        pass

    def predict(self, df):
        pass

    def save_models(self, encoder_file, decoder_file):
        pass


class Adage(AutoEncoderBase):
    def __init__(self, original_dim, latent_dim,
                 batch_size, epochs, learning_rate, sparsity, noise, epsilon_std):
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.sparsity = sparsity
        self.noise = noise
        self.epsilon_std = epsilon_std


    def build_encoder_layer(self):
        self.input_rnaseq = Input(shape=(self.original_dim,))
        encoded_rnaseq = Dropout(self.noise)(self.input_rnaseq)
        encoded_rnaseq_2 = Dense(self.latent_dim,
                                 activity_regularizer=l1(self.sparsity))(encoded_rnaseq)
        self.activation = Activation('relu')(encoded_rnaseq_2)
        self.encoder = Model(self.input_rnaseq, encoded_rnaseq_2)

    def build_decoder_layer(self):
        self.rnaseq_reconstruct = Dense(self.original_dim, activation='sigmoid')(self.activation)
        self.vae = Model(self.input_rnaseq, self.rnaseq_reconstruct)

        encoded_input = Input(shape=(self.latent_dim,))
        decoder_layer = self.vae.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))

    def compile_vae(self):
        adadelta = optimizers.Adadelta(lr=self.learning_rate)
        self.vae.compile(optimizer=adadelta, loss='mse')

    def get_summary(self):
        self.vae.summary()

    def train_vae(self, rnaseq_train_df, rnaseq_test_df, boardlog_path):
        self.hist = self.vae.fit(np.array(rnaseq_train_df),np.array(rnaseq_train_df),
                                 shuffle=True,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(np.array(rnaseq_test_df), np.array(rnaseq_test_df)),
                                 callbacks=[TensorBoard(log_dir=boardlog_path)])

    def visualize_training(self, output_file):
        # Visualize training performance
        history_df = pd.DataFrame(self.hist.history)
        ax = history_df.plot()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('VAE Loss')
        fig = ax.get_figure()
        fig.savefig(output_file)

    def compress(self, df):
        self.encoded_samples = self.encoder.predict(np.array(df))
        encoded_rnaseq_df = pd.DataFrame(self.encoded_samples, index=df.index)
        return encoded_rnaseq_df

    def get_decoder_weights(self, df):
        # Output weight matrix of gene contributions per node
        weight_file = os.path.join('results', 'adage_gene_weights.tsv')

        weight_matrix = pd.DataFrame(self.vae.get_weights()[0], index=df.columns,
                                     columns=range(1, 101)).T
        weight_matrix.index.name = 'encodings'
        weight_matrix.to_csv(weight_file, sep='\t')

    def predict(self, df):
        return self.decoder.predict(np.array(df))

    def save_models(self, encoder_file, decoder_file):
        self.encoder.save(encoder_file)
        self.decoder.save(decoder_file)


class Tybalt(AutoEncoderBase):
    """
    Facilitates the training and output of tybalt model trained on TCGA RNAseq gene expression data
    """

    def __init__(self, original_dim, hidden_dim, latent_dim,
                 batch_size, epochs, learning_rate, kappa, epsilon_std, depth):
        self.original_dim = original_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.kappa = kappa
        self.beta = backend.variable(0)
        self.epsilon_std = epsilon_std
        self.depth = depth

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = backend.random_normal(shape=(backend.shape(z_mean)[0], self.latent_dim), mean=0.,
                                  stddev=self.epsilon_std)
        return z_mean + backend.exp(z_log_var / 2) * epsilon

    def build_encoder_layer(self):
        # Input place holder for RNAseq data with specific input size
        self.rnaseq_input = Input(shape=(self.original_dim,))

        if self.depth == 1:
            z_mean_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(self.rnaseq_input)
            z_log_var_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(self.rnaseq_input)

        else:
            # depth = 2
            hidden_dense_linear = Dense(self.hidden_dim, kernel_initializer='glorot_uniform')(self.rnaseq_input)
            hidden_dense_batchnorm = BatchNormalization()(hidden_dense_linear)
            hidden_encoded = Activation('relu')(hidden_dense_batchnorm)
            z_mean_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(hidden_encoded)
            z_log_var_dense_linear = Dense(self.latent_dim, kernel_initializer='glorot_uniform')(hidden_encoded)

        z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
        self.z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)

        z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
        self.z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)

        self.z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean_encoded, self.z_log_var_encoded])

    def build_decoder_layer(self):
        if self.depth == 1:
            self.decoder_model = Dense(self.original_dim, kernel_initializer='glorot_uniform', activation='sigmoid')
        elif self.depth == 2:
            self.decoder_model = Sequential()
            self.decoder_model.add(Dense(self.hidden_dim, activation='relu', input_dim=self.latent_dim))
            self.decoder_model.add(Dense(self.original_dim, activation='sigmoid'))

        self.rnaseq_reconstruct = self.decoder_model(self.z)

    def compile_vae(self):
        adam = optimizers.Adam(lr=self.learning_rate)
        vae_layer = CustomVariationalLayer(self.original_dim, self.z_log_var_encoded,
                                           self.z_mean_encoded, self.beta)([self.rnaseq_input, self.rnaseq_reconstruct])
        self.vae = Model(self.rnaseq_input, vae_layer)
        self.vae.compile(optimizer=adam, loss=None, loss_weights=[self.beta])

    def get_summary(self):
        self.vae.summary()

    # def visualize_architecture(self, output_file):
    #     # Visualize the connections of the custom VAE model
    #     plot_model(self.vae, to_file=output_file)
    #     SVG(model_to_dot(self.vae).create(prog='dot', format='svg'))

    def train_vae(self, rnaseq_train_df, rnaseq_test_df, board_log_path):
        self.hist = self.vae.fit(np.array(rnaseq_train_df),
                                 shuffle=True,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(np.array(rnaseq_test_df), np.array(rnaseq_test_df)),
                                 callbacks=[WarmUpCallback(self.beta, self.kappa),
                                            TensorBoard(log_dir=board_log_path)])

    def visualize_training(self, output_file):
        # Visualize training performance
        history_df = pd.DataFrame(self.hist.history)
        ax = history_df.plot()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('VAE Loss')
        fig = ax.get_figure()
        fig.savefig(output_file)

    def compress(self, df):
        # Model to compress input
        self.encoder = Model(self.rnaseq_input, self.z_mean_encoded)

        # Encode rnaseq into the hidden/latent representation - and save output
        encoded_df = self.encoder.predict_on_batch(df)
        encoded_df = pd.DataFrame(encoded_df, columns=range(1, self.latent_dim + 1),
                                  index=df.index)
        return encoded_df

    def get_decoder_weights(self, df):
        # build a generator that can sample from the learned distribution
        decoder_input = Input(shape=(self.latent_dim,))  # can generate from any sampled z vector
        _x_decoded_mean = self.decoder_model(decoder_input)
        self.decoder = Model(decoder_input, _x_decoded_mean)
        weights = []
        for layer in self.decoder.layers:
            weights.append(layer.get_weights())
        return (weights)

    def predict(self, df):
        return self.decoder.predict(np.array(df))

    def save_models(self, encoder_file, decoder_file):
        self.encoder.save(encoder_file)
        self.decoder.save(decoder_file)