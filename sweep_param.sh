#!/usr/bin/env bash

echo 最適なハイパーパラメータの探索
~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python sweep_param.py --parameter_file 'config/parameter_sweep.tsv' --python_path '~/.pyenv/versions/anaconda3-4.4.0/envs/geainer/bin/python' --param_folder 'sweep_param'
