#!/usr/bin/env bash

echo 最適なハイパーパラメータの探索 Tybalt
~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python sweep_param.py --cancer_type pancan --parameter_file 'config/tybalt_param_sweep.tsv' --python_path '~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python'

echo 最適なハイパーパラメータの探索 Adage
~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python sweep_param.py --cancer_type pancan --parameter_file 'config/adage_param_sweep.tsv' --python_path '~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python'

