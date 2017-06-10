#!/bin/bash

path_dataset='/data/anowak/QAP/'
path_logger='/home/anowak/tmp/QAP1/'

mkdir -p $path_dataset
mkdir -p $path_logger

python main.py --path_dataset $path_dataset --path_logger $path_logger