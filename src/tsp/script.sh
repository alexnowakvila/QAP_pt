#!/bin/bash

path_dataset='/data/anowak/TSP/'
path_logger='/home/anowak/tmp/TSP1/'
path_tsp='/home/anowak/QAP_pt/src/tsp/LKH/'

mkdir -p $path_dataset
mkdir -p $path_logger

python main.py --path_dataset $path_dataset --path_logger $path_logger \
               --path_tsp $path_tsp --clip_grad_norm 2.0