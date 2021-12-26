#!/bin/bash

export PATH=/usr/local.orig/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local.orig/cuda-10.0/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
. "/public/zhiyuan/anaconda3/etc/profile.d/conda.sh"

conda activate torch
python3 param_search.py --model model_v1
