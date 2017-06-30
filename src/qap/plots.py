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
    Noise = np.array([0.0, 0.005, 0.01, 0.015, 0.02, 0.025,0.030,0.040, 0.050])
    SDP_ErdosRenyi_means = [1, 1, 1, 1, 1 ,0.98, 1, 0.96, 0.94]
    SDP_ErdosRenyi_stds = [0, 0, 0, 0, 0, 0.03, 0, 0.06, 0.08]
    LowRank_ErdosRenyi_means = [1, 0.86, 0.84, 0.76, 0.66, 0.66, 0.5, 
                                0.35, 0.30]
    LowRank_ErdosRenyi_stds = [0, 0.15, 0.15, 0.15, 0.25, 0.25, 0.3,
                               0.3, 0.3]
    SDP_Regular_means = [0.08, 0.06, 0.04, 0.03, 0.02, 0.03, 0.05, 0.02,0.03]
    SDP_Regular_stds = [0.08, 0.04,0.04,0.02,0.02,0.02,0.02,0.02,0.01]
    LowRank_Regular_means = [1, 0.83, 0.65, 0.4, 0.42,0.3,0.34,0.14,0.14]
    LowRank_Regular_stds = [0,0.25,0.25,0.25,0.25,0.25,0.25,0.15,0.20]
    filename = 'results.npz'
    Names = ['ErdosRenyi Graph Model', 'Random Regular Graph Model']
    Models = ['ErdosRenyi', 'Regular']
    Comparatives_ErdosRenyi = [SDP_ErdosRenyi_means, SDP_ErdosRenyi_stds,
                               LowRank_ErdosRenyi_means,
                               LowRank_ErdosRenyi_stds]
    Comparatives_Regular = [SDP_Regular_means, SDP_Regular_stds,
                            LowRank_Regular_means,
                            LowRank_Regular_stds]
    Comparatives = {'ErdosRenyi':Comparatives_ErdosRenyi,
                    'Regular':Comparatives_Regular}
    main_path = '/home/anowak/tmp/'
    for j, model in enumerate(Models):
        Dire = ['/home/anowak/tmp/QAP0.000' + model + '_3/',
                '/home/anowak/tmp/QAP0.005' + model + '_3/',
                '/home/anowak/tmp/QAP0.010' + model + '_3/',
                '/home/anowak/tmp/QAP0.015' + model + '_3/',
                '/home/anowak/tmp/QAP0.020' + model + '_3/',
                '/home/anowak/tmp/QAP0.025' + model + '_3/',
                '/home/anowak/tmp/QAP0.030' + model + '_3/',
                '/home/anowak/tmp/QAP0.040' + model + '_3/',
                '/home/anowak/tmp/QAP0.050' + model + '_3/',]
        ErrorBars = []
        AccuracyMean = []
        H = []
        L = []
        for k, dire in enumerate(Dire):
            path = os.path.join(dire, filename)
            npz = np.load(path)
            accuracy = npz['accuracy_train']
            # print(accuracy.shape)
            accuracy_mean = []
            for i in range(accuracy.shape[0]-100):
                std = accuracy[i:i+100].std()
                accuracy_mean.append(accuracy[i:i+100].mean())
            ErrorBars.append(std)
            AccuracyMean.append(accuracy_mean[-1])
            # print(accuracy_mean)
            plt.figure(0)
            plt.clf()
            plt.plot(accuracy_mean, c='r')
            plt.savefig(dire + 'accuracy_mean.png')
        ErrorBars = np.array(ErrorBars)
        AccuracyMean = np.array(AccuracyMean)
        fig = plt.figure(1)
        plt.clf()
        # plt.subplot(1, 2, j + 1)
        # SDP
        SDP = Comparatives[model][:2]
        LowRank = Comparatives[model][2:]
        print(len(SDP[0]), len(SDP[1]))
        plt.errorbar(Noise, SDP[0], yerr=[SDP[1], SDP[1]],
                     fmt='-o', c='b', label='SDP')
        plt.errorbar(Noise, LowRank[0], yerr=[LowRank[1], LowRank[1]],
                     fmt='-o', c='g', label='LowRankAlign(k=4)')
        plt.errorbar(Noise, AccuracyMean, yerr=[ErrorBars, ErrorBars],
                     fmt='-o', c='r', label='GNN')
        # h1, l1 = fig.get_legend_handles_labels()
        plt.xlabel('Noise')
        plt.ylabel('Recovery Rate')
        plt.title(Names[j], fontsize=25)
        # l = [lsdp, llrk, lres]
        # names ['SDP', 'LowRank', 'GNN']
        if j == 0:
            plt.legend(loc='lower left', prop={'size':12})
        plt.savefig(main_path + model + '.eps', format='eps')
    # plt.tight_layout(pad=0.4, w_pad=2.0, h_pad=1.0)
    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #       fancybox=True, shadow=True, ncol=5)
    # plt.savefig(main_path + 'QAP_results.png')