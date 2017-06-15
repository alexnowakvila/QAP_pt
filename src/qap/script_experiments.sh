#!/bin/bash

path_dataset='/data/anowak/QAP/'
path_logger='/home/anowak/tmp/'

mkdir -p $path_dataset
mkdir -p $path_logger

###############################################################################
#                                FIRST SET                                    #
###############################################################################

# erdosrenyi='ErdosRenyi'
# regular='Regular'
# QAP='QAP'
# bar='/'
# Noise_levels='0.000 0.005 0.010 0.015 0.020 0.025'
# for noise in $Noise_levels
# do
# path_logs=$path_logger$QAP$noise$bar
# mkdir -p $path_logs
# python main.py --path_dataset $path_dataset --path_logger $path_logs \
#                --generative_model $erdosrenyi --noise $noise 
# done
# echo All done

###############################################################################
#                               SECOND SET                                    #
###############################################################################

# erdosrenyi='ErdosRenyi'
# regular='Regular'
# QAP='QAP'
# bar='/'
# Noise_levels='0.030 0.035 0.040 0.045 0.050'
# for noise in $Noise_levels
# do
# path_logs=$path_logger$QAP$noise$bar
# mkdir -p $path_logs
# python main.py --path_dataset $path_dataset --path_logger $path_logs \
#                --generative_model $erdosrenyi --noise $noise
# done
# echo All done

###############################################################################
#                                THIRD SET                                    #
###############################################################################

erdosrenyi='ErdosRenyi'
regular='Regular'
QAP='QAP'
bar='/'
Noise_levels='0.000 0.005 0.010 0.015 0.020 0.025'
for noise in $Noise_levels
do
path_logs=$path_logger$QAP$noise$bar
mkdir -p $path_logs
python main.py --path_dataset $path_dataset --path_logger $path_logs \
               --generative_model $regular --noise $noise
done
echo All done

###############################################################################
#                                FORTH SET                                    #
###############################################################################

# erdosrenyi='ErdosRenyi'
# regular='Regular'
# QAP='QAP'
# bar='/'
# Noise_levels='0.030 0.035 0.040 0.045 0.050'
# for noise in $Noise_levels
# do
# path_logs=$path_logger$QAP$noise$bar
# mkdir -p $path_logs
# python main.py --path_dataset $path_dataset --path_logger $path_logs \
#                --generative_model $regular --noise $noise
# done
# echo All done