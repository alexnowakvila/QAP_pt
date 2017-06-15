import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


if __name__ == '__main__':
    filename = 'results.npz'
    model = 'ErdosRenyi'
    # model = 'Regular'
    main_path = '/home/anowak/tmp/'
    Dire = ['/home/anowak/tmp/QAP0.010' + model + '_2/',
            '/home/anowak/tmp/QAP0.020' + model + '_2/',
            '/home/anowak/tmp/QAP0.030' + model + '_2/',
            '/home/anowak/tmp/QAP0.040' + model + '_2/']
    Noise = np.array([0.010, 0.020, 0.030, 0.040])
    ErrorBars = []
    AccuracyMean = []
    for i, dire in enumerate(Dire):
        path = os.path.join(dire, filename)
        npz = np.load(path)
        accuracy = npz['accuracy_train']
        # print(accuracy.shape)
        accuracy_mean = []
        for i in range(accuracy.shape[0]-100):
            minimum = accuracy[i:i+100].min()
            maximum = accuracy[i:i+100].max()
            accuracy_mean.append(accuracy[i:i+100].mean())
        ErrorBars.append([accuracy_mean[-1]-minimum,
                          maximum-accuracy_mean[-1]])
        AccuracyMean.append(accuracy_mean[-1])
        # print(accuracy_mean)
        plt.figure(0)
        plt.clf()
        plt.plot(accuracy_mean, c='r')
        plt.savefig(dire + 'accuracy_mean.png')
    ErrorBars = np.array(ErrorBars)
    AccuracyMean = np.array(AccuracyMean)
    plt.figure(0)
    plt.clf()
    plt.errorbar(Noise, AccuracyMean, yerr=[ErrorBars[:, 0], ErrorBars[:, 1]],
                 fmt='-o')
    plt.xlabel('Noise')
    plt.ylabel('Recovery Rate')
    plt.title(model)
    plt.savefig(main_path + model + '.png')