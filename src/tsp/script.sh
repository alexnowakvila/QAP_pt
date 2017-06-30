#!/bin/bash

# path_logger='/home/anowak/tmp/TSP1/'
# path_logger='/home/anowak/tmp/TSP3/'
# path_logger='/home/anowak/tmp/TSP9/'
path_logger='/home/anowak/tmp/TSP_deep50/'
path_load='/home/anowak/tmp/TSPprova/'
path_dataset='/data/anowak/TSP/'
path_tsp='/home/anowak/QAP_pt/src/tsp/LKH/'

mkdir -p $path_dataset
mkdir -p $path_logger

# python main.py --path_dataset $path_dataset --path_logger $path_logger \
#                --path_tsp $path_tsp --clip_grad_norm 40.0 --beam_size 40 \
#                --batch_size 1 --num_features 120 --num_layers 40 \
#                --num_examples_train 3000 --num_examples_test 100 --dual \

# python main.py --path_dataset $path_dataset --path_logger $path_logger \
#                --path_tsp $path_tsp --clip_grad_norm 40.0 --beam_size 40 \
#                --batch_size 16 --num_features 120 --num_layers 40 \

# python main.py --path_dataset $path_dataset --path_logger $path_logger \
#                --path_tsp $path_tsp --clip_grad_norm 40.0 --beam_size 40 \
#                --batch_size 64 --num_features 120 --num_layers 40 \
#                --path_load $path_load --load \

# python main.py --path_dataset $path_dataset --path_logger $path_logger \
#                --path_tsp $path_tsp --clip_grad_norm 40.0 --beam_size 40 \
#                --batch_size 32 --num_features 120 --num_layers 40 \
#                --load --path_load $path_load

python main.py --path_dataset $path_dataset --path_logger $path_logger \
               --path_tsp $path_tsp --clip_grad_norm 40.0 --beam_size 40 \
               --batch_size 32 --num_features 80 --num_layers 60 --N 50 \
