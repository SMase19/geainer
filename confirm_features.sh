#!/usr/bin/env bash

echo 特徴抽出開始
python utils/tsne_tybalt_features.py
echo 可視化ファイル作成
Rscript utils/tsne_viz.R
open figures/tsne_rnaseq.pdf
open figures/tsne_vae.pdf
