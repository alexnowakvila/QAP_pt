#!/bin/bash

path_dataset='/data/anowak/QAP/'
path_logger='/home/anowak/tmp/'

QAP='QAP'
noise='0.030'
# noise='RN'
bar='_3/'
mkdir -p $path_dataset
mkdir -p $path_logger
model='ErdosRenyi'
# model='Regular'
path_logs=$path_logger$QAP$noise$model$bar
mkdir -p $path_logs

python main.py --path_dataset $path_dataset --path_logger $path_logs \
               --generative_model $model --noise $noise --num_features 20 \
#               --generative_model $model --random_noise --num_features 20 \
#               --generative_model $model --noise $noise --num_features 20 \
               