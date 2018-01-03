#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
# import dependencies
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import networkx

#Pytorch requirements
import unicodedata
import string
import re
import random
import argparse

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class Generator(object):
    def __init__(self, path_dataset):
        self.path_dataset = path_dataset
        self.num_examples_train = 10e6
        self.num_examples_test = 10e4
        self.data_train = []
        self.data_test = []
        self.J = 3
        self.N = 50
        self.generative_model = 'ErdosRenyi'
        self.edge_density = 0.2
        self.random_noise = False
        self.noise = 0.03
        self.noise_model = 2

    def ErdosRenyi(self, p, N):
        W = np.zeros((N, N))
        for i in range(0, N - 1):
            for j in range(i + 1, N):
                add_edge = (np.random.uniform(0, 1) < p)
                if add_edge:
                    W[i, j] = 1
                W[j, i] = W[i, j]
        return W

    def ErdosRenyi_netx(self, p, N):
        g = networkx.erdos_renyi_graph(N, p)
        W = networkx.adjacency_matrix(g).todense().astype(float)
        W = np.array(W)
        return W

    def RegularGraph_netx(self, p, N):
        """ Generate random regular graph """
        d = p * N
        d = int(d)
        g = networkx.random_regular_graph(d, N)
        W = networkx.adjacency_matrix(g).todense().astype(float)
        W = np.array(W)
        return W

    def compute_operators(self, W):
        N = W.shape[0]
        if self.generative_model == 'ErdosRenyi':
            # operators: {Id, W, W^2, ..., W^{J-1}, D, U}
            d = W.sum(1)
            D = np.diag(d)
            QQ = W.copy()
            WW = np.zeros([N, N, self.J + 2])
            WW[:, :, 0] = np.eye(N)
            for j in range(self.J):
                WW[:, :, j + 1] = QQ.copy()
                # QQ = np.dot(QQ, QQ)
                QQ = np.minimum(np.dot(QQ, QQ), np.ones(QQ.shape))
            WW[:, :, self.J] = D
            WW[:, :, self.J + 1] = np.ones((N, N)) * 1.0 / float(N)
            WW = np.reshape(WW, [N, N, self.J + 2])
            x = np.reshape(d, [N, 1])
        elif self.generative_model == 'Regular':
            # operators: {Id, A, A^2}
            ds = []
            ds.append(W.sum(1))
            QQ = W.copy()
            WW = np.zeros([N, N, self.J + 2])
            WW[:, :, 0] = np.eye(N)
            for j in range(self.J):
                WW[:, :, j + 1] = QQ.copy()
                # QQ = np.dot(QQ, QQ)
                QQ = np.minimum(np.dot(QQ, QQ), np.ones(QQ.shape))
                ds.append(QQ.sum(1))
            d = ds[1]
            D = np.diag(ds[1])
            WW[:, :, self.J] = D
            WW[:, :, self.J + 1] = np.ones((N, N)) * 1.0 / float(N)
            WW = np.reshape(WW, [N, N, self.J + 2])
            x = np.reshape(d, [N, 1])
        else:
            raise ValueError('Generative model {} not implemented'
                             .format(self.generative_model))
        return WW, x

    def compute_example(self):
        example = {}
        if self.generative_model == 'ErdosRenyi':
            W = self.ErdosRenyi_netx(self.edge_density, self.N)
        elif self.generative_model == 'Regular':
            W = self.RegularGraph_netx(self.edge_density, self.N)
        else:
            raise ValueError('Generative model {} not supported'
                             .format(self.generative_model))
        if self.random_noise:
            self.noise = np.random.uniform(0.000, 0.050, 1)
        if self.noise_model == 1:
            # use noise model from [arxiv 1602.04181], eq (3.8)
            noise = self.ErdosRenyi(self.noise, self.N)
            W_noise = W*(1-noise) + (1-W)*noise
        elif self.noise_model == 2:
            # use noise model from [arxiv 1602.04181], eq (3.9)
            pe1 = self.noise
            pe2 = (self.edge_density*self.noise)/(1.0-self.edge_density)
            noise1 = self.ErdosRenyi_netx(pe1, self.N)
            noise2 = self.ErdosRenyi_netx(pe2, self.N)
            W_noise = W*(1-noise1) + (1-W)*noise2
        else:
            raise ValueError('Noise model {} not implemented'
                             .format(self.noise_model))
        WW, x = self.compute_operators(W)
        WW_noise, x_noise = self.compute_operators(W_noise)
        example['WW'], example['x'] = WW, x
        example['WW_noise'], example['x_noise'] = WW_noise, x_noise
        return example

    def create_dataset_train(self):
        for i in range(self.num_examples_train):
            example = self.compute_example()
            self.data_train.append(example)

    def create_dataset_test(self):
        for i in range(self.num_examples_test):
            example = self.compute_example()
            self.data_test.append(example)

    def load_dataset(self):
        # load train dataset
        if self.random_noise:
            filename = 'QAPtrain_RN.np'
        else:
            filename = ('QAPtrain_{}_{}_{}.np'.format(self.generative_model,
                        self.noise, self.edge_density))
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path):
            print('Reading training dataset at {}'.format(path))
            self.data_train = np.load(open(path, 'rb'))
        else:
            print('Creating training dataset.')
            self.create_dataset_train()
            print('Saving training datatset at {}'.format(path))
            np.save(open(path, 'wb'), self.data_train)
        # load test dataset
        if self.random_noise:
            filename = 'QAPtest_RN.np'
        else:
            filename = ('QAPtest_{}_{}_{}.np'.format(self.generative_model,
                        self.noise, self.edge_density))
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path):
            print('Reading testing dataset at {}'.format(path))
            self.data_test = np.load(open(path, 'rb'))
        else:
            print('Creating testing dataset.')
            self.create_dataset_test()
            print('Saving testing datatset at {}'.format(path))
            np.save(open(path, 'wb'), self.data_test)

    def sample_batch(self, num_samples, is_training=True,
                     cuda=True, volatile=False):
        WW_size = self.data_train[0]['WW'].shape
        x_size = self.data_train[0]['x'].shape

        WW = torch.zeros(WW_size).expand(num_samples, *WW_size)
        X = torch.zeros(x_size).expand(num_samples, *x_size)
        WW_noise = torch.zeros(WW_size).expand(num_samples, *WW_size)
        X_noise = torch.zeros(x_size).expand(num_samples, *x_size)

        if is_training:
            dataset = self.data_train
        else:
            datatset = self.data_test
        for b in range(num_samples):
            if is_training:
                ind = np.random.randint(0, len(dataset))
            else:
                ind = b
            ww = torch.from_numpy(dataset[ind]['WW'])
            x = torch.from_numpy(dataset[ind]['x'])
            WW[b] = ww
            X[b] = x
            ww_noise = torch.from_numpy(dataset[ind]['WW_noise'])
            x_noise = torch.from_numpy(dataset[ind]['x_noise'])
            WW_noise[b] = ww_noise
            X_noise[b] = x_noise
        
        WW = Variable(WW, volatile=volatile)
        X = Variable(X, volatile=volatile)
        WW_noise = Variable(WW_noise, volatile=volatile)
        X_noise = Variable(X_noise, volatile=volatile)

        if cuda:
            return [WW.cuda(), X.cuda()], [WW_noise.cuda(), X_noise.cuda()]
        else:
            return [WW, X], [WW_noise, X_noise]

if __name__ == '__main__':
    ###################### Test Generator module ##############################
    path = '/home/anowak/tmp/'
    gen = Generator(path)
    gen.num_examples_train = 10
    gen.num_examples_test = 10
    gen.N = 50
    gen.generative_model = 'Regular'
    gen.load_dataset()
    g1, g2 = gen.sample_batch(32, cuda=False)
    print(g1[0].size())
    print(g1[1][0].data.cpu().numpy())
    W = g1[0][0, :, :, 1]
    W_noise = g2[0][0, :, :, 1]
    print(W, W.size())
    print(W_noise.size(), W_noise)
    ################### Test graph generators networkx ########################
    # path = '/home/anowak/tmp/'
    # gen = Generator(path)
    # p = 0.2
    # N = 50
    # # W = gen.ErdosRenyi_netx(p, N)
    # W = gen.RegularGraph_netx(3, N)
    # G = networkx.from_numpy_matrix(W)
    # networkx.draw(G)
    # # plt.draw(G)
    # plt.savefig('/home/anowak/tmp/prova.png')
    # print('W', W)