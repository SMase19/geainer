#!/usr/bin/env bash
#
## tybalt lung
#~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python run_vae.py --algorithm tybalt --cancer_type lung --learning_rate 0.0005 --batch_size 50 --epochs 50 --kappa 1 --depth 1 --first_layer 100 --output_board_log data/lung/tybalt/logs/learning=0.0005,batch=50,epochs=50,kappa=1,depth=1,first=100
#bash confirm_features.sh
#mv data/encoded_rnaseq.tsv data/lung/tybalt/logs/learning=0.0005,batch=50,epochs=50,kappa=1,depth=1,first=100
#mv results/tsne_features.tsv data/lung/tybalt/logs/learning=0.0005,batch=50,epochs=50,kappa=1,depth=1,first=100
#mv figures/tsne_vae.pdf data/lung/tybalt/logs/learning=0.0005,batch=50,epochs=50,kappa=1,depth=1,first=100
#mv figures/tsne_vae.png data/lung/tybalt/logs/learning=0.0005,batch=50,epochs=50,kappa=1,depth=1,first=100
#mv models/decoder_vae.hdf5 data/lung/tybalt/logs/learning=0.0005,batch=50,epochs=50,kappa=1,depth=1,first=100
#mv models/encoder_vae.hdf5 data/lung/tybalt/logs/learning=0.0005,batch=50,epochs=50,kappa=1,depth=1,first=100
#
## adage lung
#~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python run_vae.py --algorithm adage --cancer_type lung --learning_rate 1.1 --batch_size 50 --epochs 100 --sparsity 0 --noise 0.05 --output_board_log data/lung/adage/logs/learning=1.1,batch=50,epochs=100,sparsity=0,noise=0.05
#bash confirm_features.sh
#mv data/encoded_rnaseq.tsv data/lung/adage/logs/learning=1.1,batch=50,epochs=100,sparsity=0,noise=0.05
#mv results/tsne_features.tsv data/lung/adage/logs/learning=1.1,batch=50,epochs=100,sparsity=0,noise=0.05
#mv figures/tsne_vae.pdf data/lung/adage/logs/learning=1.1,batch=50,epochs=100,sparsity=0,noise=0.05
#mv figures/tsne_vae.png data/lung/adage/logs/learning=1.1,batch=50,epochs=100,sparsity=0,noise=0.05
#mv models/decoder_vae.hdf5 data/lung/adage/logs/learning=1.1,batch=50,epochs=100,sparsity=0,noise=0.05
#mv models/encoder_vae.hdf5 data/lung/adage/logs/learning=1.1,batch=50,epochs=100,sparsity=0,noise=0.05
#
## tybalt lung latent 100 100
#~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python run_vae.py --algorithm tybalt --cancer_type lung --learning_rate 0.001 --batch_size 100 --epochs 100 --kappa 1 --depth 2 --first_layer 100 --output_board_log data/lung/tybalt/logs/learning=0.001,batch=100,epochs=100,kappa=1,depth=2,first=100
#bash confirm_features.sh
#mv data/encoded_rnaseq.tsv data/lung/tybalt/logs/learning=0.001,batch=100,epochs=100,kappa=1,depth=2,first=100
#mv results/tsne_features.tsv data/lung/tybalt/logs/learning=0.001,batch=100,epochs=100,kappa=1,depth=2,first=100
#mv figures/tsne_vae.pdf data/lung/tybalt/logs/learning=0.001,batch=100,epochs=100,kappa=1,depth=2,first=100
#mv figures/tsne_vae.png data/lung/tybalt/logs/learning=0.001,batch=100,epochs=100,kappa=1,depth=2,first=100
#mv models/decoder_vae.hdf5 data/lung/tybalt/logs/learning=0.001,batch=100,epochs=100,kappa=1,depth=2,first=100
#mv models/encoder_vae.hdf5 data/lung/tybalt/logs/learning=0.001,batch=100,epochs=100,kappa=1,depth=2,first=100
#
## tybalt lung latent 100 300
#~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python run_vae.py --algorithm tybalt --cancer_type lung --learning_rate 0.0005 --batch_size 50 --epochs 100 --kappa 0.01 --depth 2 --first_layer 300 --output_board_log data/lung/tybalt/logs/learning=0.0005,batch=50,epochs=100,kappa=0.01,depth=2,first=300
#bash confirm_features.sh
#mv data/encoded_rnaseq.tsv data/lung/tybalt/logs/learning=0.0005,batch=50,epochs=100,kappa=0.01,depth=2,first=300
#mv results/tsne_features.tsv data/lung/tybalt/logs/learning=0.0005,batch=50,epochs=100,kappa=0.01,depth=2,first=300
#mv figures/tsne_vae.pdf data/lung/tybalt/logs/learning=0.0005,batch=50,epochs=100,kappa=0.01,depth=2,first=300
#mv figures/tsne_vae.png data/lung/tybalt/logs/learning=0.0005,batch=50,epochs=100,kappa=0.01,depth=2,first=300
#mv models/decoder_vae.hdf5 data/lung/tybalt/logs/learning=0.0005,batch=50,epochs=100,kappa=0.01,depth=2,first=300
#mv models/encoder_vae.hdf5 data/lung/tybalt/logs/learning=0.0005,batch=50,epochs=100,kappa=0.01,depth=2,first=300
#
## tybalt pancan
#~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python run_vae.py --algorithm tybalt --cancer_type pancan --learning_rate 0.0005 --batch_size 50 --epochs 50 --kappa 1 --depth 1 --first_layer 100 --output_board_log data/pancan/tybalt/logs/learning=0.0005,batch=50,epochs=50,kappa=1,depth=1,first=100
#bash confirm_features.sh
#mv data/encoded_rnaseq.tsv data/pancan/tybalt/logs/learning=0.0005,batch=50,epochs=50,kappa=1,depth=1,first=100
#mv results/tsne_features.tsv data/pancan/tybalt/logs/learning=0.0005,batch=50,epochs=50,kappa=1,depth=1,first=100
#mv figures/tsne_vae.pdf data/pancan/tybalt/logs/learning=0.0005,batch=50,epochs=50,kappa=1,depth=1,first=100
#mv figures/tsne_vae.png data/pancan/tybalt/logs/learning=0.0005,batch=50,epochs=50,kappa=1,depth=1,first=100
#mv models/decoder_vae.hdf5 data/pancan/tybalt/logs/learning=0.0005,batch=50,epochs=50,kappa=1,depth=1,first=100
#mv models/encoder_vae.hdf5 data/pancan/tybalt/logs/learning=0.0005,batch=50,epochs=50,kappa=1,depth=1,first=100
#
## adage pancan
#~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python run_vae.py --algorithm adage --cancer_type pancan --learning_rate 1.1 --batch_size 50 --epochs 100 --sparsity 0 --noise 0.05 --output_board_log data/pancan/adage/logs/learning=1.1,batch=50,epochs=100,sparsity=0,noise=0.05
#bash confirm_features.sh
#mv data/encoded_rnaseq.tsv data/pancan/adage/logs/learning=1.1,batch=50,epochs=100,sparsity=0,noise=0.05
#mv results/tsne_features.tsv data/pancan/adage/logs/learning=1.1,batch=50,epochs=100,sparsity=0,noise=0.05
#mv figures/tsne_vae.pdf data/pancan/adage/logs/learning=1.1,batch=50,epochs=100,sparsity=0,noise=0.05
#mv figures/tsne_vae.png data/pancan/adage/logs/learning=1.1,batch=50,epochs=100,sparsity=0,noise=0.05
#mv models/decoder_vae.hdf5 data/pancan/adage/logs/learning=1.1,batch=50,epochs=100,sparsity=0,noise=0.05
#mv models/encoder_vae.hdf5 data/pancan/adage/logs/learning=1.1,batch=50,epochs=100,sparsity=0,noise=0.05

# tybalt pancan latent 100 100
~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python run_vae.py --algorithm tybalt --cancer_type pancan --learning_rate 0.001 --batch_size 100 --epochs 100 --kappa 1 --depth 2 --first_layer 100 --output_board_log data/pancan/tybalt/logs/learning=0.001,batch=100,epochs=100,kappa=1,depth=2,first=100
bash confirm_features.sh
mv data/encoded_rnaseq.tsv data/pancan/tybalt/logs/learning=0.001,batch=100,epochs=100,kappa=1,depth=2,first=100
mv results/tsne_features.tsv data/pancan/tybalt/logs/learning=0.001,batch=100,epochs=100,kappa=1,depth=2,first=100
mv figures/tsne_vae.pdf data/pancan/tybalt/logs/learning=0.001,batch=100,epochs=100,kappa=1,depth=2,first=100
mv figures/tsne_vae.png data/pancan/tybalt/logs/learning=0.001,batch=100,epochs=100,kappa=1,depth=2,first=100
mv models/decoder_vae.hdf5 data/pancan/tybalt/logs/learning=0.001,batch=100,epochs=100,kappa=1,depth=2,first=100
mv models/encoder_vae.hdf5 data/pancan/tybalt/logs/learning=0.001,batch=100,epochs=100,kappa=1,depth=2,first=100
#
## tybalt pancan latent 100 300
#~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python run_vae.py --algorithm tybalt --cancer_type pancan --learning_rate 0.0005 --batch_size 50 --epochs 100 --kappa 0.01 --depth 2 --first_layer 300 --output_board_log data/pancan/tybalt/logs/learning=0.0005,batch=50,epochs=100,kappa=0.01,depth=2,first=300
#bash confirm_features.sh
#mv data/encoded_rnaseq.tsv data/pancan/tybalt/logs/learning=0.0005,batch=50,epochs=100,kappa=0.01,depth=2,first=300
#mv results/tsne_features.tsv data/pancan/tybalt/logs/learning=0.0005,batch=50,epochs=100,kappa=0.01,depth=2,first=300
#mv figures/tsne_vae.pdf data/pancan/tybalt/logs/learning=0.0005,batch=50,epochs=100,kappa=0.01,depth=2,first=300
#mv figures/tsne_vae.png data/pancan/tybalt/logs/learning=0.0005,batch=50,epochs=100,kappa=0.01,depth=2,first=300
#mv models/decoder_vae.hdf5 data/pancan/tybalt/logs/learning=0.0005,batch=50,epochs=100,kappa=0.01,depth=2,first=300
#mv models/encoder_vae.hdf5 data/pancan/tybalt/logs/learning=0.0005,batch=50,epochs=100,kappa=0.01,depth=2,first=300
