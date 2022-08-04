#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 573-env

python NER_train.py
