#!/usr/bin/env python
"""Chainer example: train a VAE on MNIST
"""
from __future__ import print_function
import argparse, os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import six

import chainer
from chainer import computational_graph as c
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import functions as f

# import data
import ae_base as ae


parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=50, type=int,
                    help='number of epochs to learn')
parser.add_argument('--dimz', '-z', default=100, type=int,
                    help='dimention of encoded vector')
parser.add_argument('--batchsize', '-b', type=int, default=50,
                    help='learning minibatch size')
args = parser.parse_args()

batch_size = args.batchsize
n_epoch = args.epoch
n_latent = args.dimz

print('GPU: {}'.format(args.gpu))
print('# dim z: {}'.format(args.dimz))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')

# Prepare dataset
print('load pancaner dataset')


np.random.seed(123)

# 遺伝子発現量RNAseqを読み込む
rnaseq_file = os.path.join('data', 'pancan_scaled_zeroone_rnaseq.tsv')
rnaseq_df = pd.read_table(rnaseq_file, index_col=0).astype(np.float32)
print(rnaseq_df.shape)
print(rnaseq_df.values[0].dtype)


# Split 10% test set randomly
test_set_percent = 0.1
rnaseq_test_df = rnaseq_df.sample(frac=test_set_percent).astype(np.float32)
rnaseq_train_df = rnaseq_df.drop(rnaseq_test_df.index).astype(np.float32)

original_dim = rnaseq_df.shape[1]
N = rnaseq_train_df.shape[0]

# Prepare VAE model
model = ae.TybaltVAE(original_dim, n_latent, n_latent, act_func=f.relu)

if args.gpu >= 0:
    cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

# Setup optimizer
optimizer = optimizers.Adam(alpha=0.0005)
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)
#
# g = c.build_computational_graph((model,), remove_split=True)  # <-- パラメタの書き方がマニュアルと違うが、これでないと動かない感じ。
# with open('./mysample.dot', 'w') as o:
#     o.write(g.dump())


# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)
    # epoch_time = time.time()

    # training
    perm = np.random.permutation(N)
    sum_loss = 0  # total loss
    sum_rec_loss = 0  # reconstruction loss
    c_value = 2
    if epoch < 3:
        c_value = epoch - 1
    for i in six.moves.range(0, N, batch_size):
        x = chainer.Variable(xp.asarray(rnaseq_train_df.values[perm[i:i + batch_size]]))

        optimizer.update(model.get_loss_func(C=c_value), x)
        sum_loss += float(model.loss.data) * len(x.data)
        sum_rec_loss += float(model.rec_loss.data) * len(x.data)

    print('train mean loss={}, mean reconstruction loss={}'
          .format(sum_loss / N, sum_rec_loss / N))

    # evaluation
    sum_loss = 0
    sum_rec_loss = 0
    with chainer.no_backprop_mode():
        for i in six.moves.range(0, rnaseq_test_df.shape[0], batch_size):
            x = chainer.Variable(xp.asarray(rnaseq_test_df.values[i:i + batch_size]))
            loss_func = model.get_loss_func()
            loss_func(x)
            sum_loss += float(model.loss.data) * len(x.data)
            sum_rec_loss += float(model.rec_loss.data) * len(x.data)
            del model.loss
    print('test  mean loss={}, mean reconstruction loss={}'
          .format(sum_loss / rnaseq_test_df.shape[0], sum_rec_loss / rnaseq_test_df.shape[0]))

# Save the model and the optimizer
print('save the model')
serializers.save_npz('mlp.model', model)
print('save the optimizer')
serializers.save_npz('mlp.state', optimizer)

model.to_cpu()


# original images and reconstructed images
def save_images(x, filename):
    fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
    for ai, xi in zip(ax.flatten(), x):
        ai.imshow(xi.reshape(28, 28))
    fig.savefig(filename)


train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
x = chainer.Variable(np.asarray(rnaseq_train_df[train_ind]))
with chainer.no_backprop_mode():
    x1 = model(x)
save_images(x.data, 'train')
save_images(x1.data, 'train_reconstructed')

test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
x = chainer.Variable(np.asarray(rnaseq_test_df[test_ind]))
with chainer.no_backprop_mode():
    x1 = model(x)
save_images(x.data, 'test')
save_images(x1.data, 'test_reconstructed')


# draw images from randomly sampled z
z = chainer.Variable(np.random.normal(0, 1, (9, n_latent)).astype(np.float32))
x = model.decode(z)
save_images(x.data, 'sampled')
