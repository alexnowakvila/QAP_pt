#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import numpy as np
import os
# import dependencies
import time
from LKH.tsp_solver import TSP
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

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

class Generator(TSP):
    def __init__(self, path_dataset, path_tsp):
        super().__init__(path_tsp)
        # TSP.__init__(self, path_dataset)
        self.path_dataset = path_dataset
        self.num_examples_train = 10e6
        self.num_examples_test = 10e4
        self.data_train = []
        self.data_test = []
        self.N = 20
        self.J = 3

    def ErdosRenyi(self, p, N):
        W = np.zeros((N, N))
        for i in range(0, N - 1):
            for j in range(i + 1, N):
                add_edge = (np.random.uniform(0, 1) < p)
                if add_edge:
                    W[i, j] = 1
                W[j, i] = W[i, j]
        return W

    def compute_operators(self, W):
        # operators: {Id, W, W^2, ..., W^{J-1}, D, U}
        N = W.shape[0]
        d = W.sum(1)
        D = np.diag(d)
        QQ = W.copy()
        WW = np.zeros([N, N, self.J + 2])
        WW[:, :, 0] = np.eye(N)
        for j in range(self.J):
            WW[:, :, j + 1] = QQ.copy()
            QQ = QQ * QQ
        WW[:, :, self.J] = D
        WW[:, :, self.J + 1] = np.ones((N, N)) * 1.0 / float(N)
        WW = np.reshape(WW, [N, N, self.J + 2])
        x = np.reshape(d, [N, 1])
        return WW, x

    def adj_from_coord(self, cities):
        N = cities.shape[0]
        W = np.zeros((N, N))
        def l2_dist(x, y):
            return np.sqrt(np.square(x - y).sum())
        for i in range(0, N - 1):
            for j in range(i + 1, N):
                W[i, j] = l2_dist(cities[i], cities[j])
                W[j, i] = W[i, j]
        return W

    def cycle_adj(self, N):
        W = np.zeros((N, N))
        W[N-1, N-2] = 1
        W[N-1, 0] = 1
        W[0, 1] = 1
        W[0, N-1] = 1
        for i in range(1, N-1):
            W[i, i-1] = 1
            W[i, i+1] = 1
        return W

    def compute_example(self):
        example = {}
        cities = self.cities_generator(self.N)
        W = self.adj_from_coord(cities)
        C = self.cycle_adj(self.N)
        WW, x = self.compute_operators(W)
        WC, x_c = self.compute_operators(C)
        example['WW'], example['x'] = WW, x
        example['WC'], example['x_c'] = WC, x_c
        # compute hamiltonian cycle
        self.save_solverformat(cities, self.N, mode='EUC_2D')
        self.tsp_solver(self.N)
        # print(cities)
        ham_cycle, length_cycle = self.extract_path(0)
        example['HAM_cycle'] = np.array(ham_cycle[:-1])
        # print(ham_cycle[:-1])
        example['Length_cycle'] = length_cycle / self.C
        return example

    def create_dataset_train(self):
        for i in range(self.num_examples_train):
            example = self.compute_example()
            self.data_train.append(example)
            if i % 100 == 0:
                print('Train example {} of length {} computed.'
                      .format(i, self.N))

    def create_dataset_test(self):
        for i in range(self.num_examples_test):
            example = self.compute_example()
            self.data_test.append(example)
            if i % 100 == 0:
                print('Test example {} of length {} computed.'
                      .format(i, self.N))

    def load_dataset(self):
        # load train dataset
        path = os.path.join(self.path_dataset, 'TSP{}train.np'.format(self.N))
        if os.path.exists(path):
            print('Reading training dataset at {}'.format(path))
            self.data_train = np.load(open(path, 'rb'))
        else:
            print('Creating training dataset.')
            self.create_dataset_train()
            print('Saving training datatset at {}'.format(path))
            np.save(open(path, 'wb'), self.data_train)
        # load test dataset
        path = os.path.join(self.path_dataset, 'TSP{}test.np'.format(self.N))
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
        #TODO: CHANGE
        WW_size = self.data_train[0]['WW'].shape
        x_size = self.data_train[0]['x'].shape

        WW = torch.zeros(WW_size).expand(num_samples, *WW_size)
        X = torch.zeros(x_size).expand(num_samples, *x_size)
        WC = torch.zeros(WW_size).expand(num_samples, *WW_size)
        X_c = torch.zeros(x_size).expand(num_samples, *x_size)

        if is_training:
            dataset = self.data_train
        else:
            datatset = self.data_test
        for b in range(num_samples):
            ind = np.random.randint(0, len(dataset))
            ww = torch.from_numpy(dataset[ind]['WW'])
            x = torch.from_numpy(dataset[ind]['x'])
            WW[b] = ww
            X[b] = x
            wc = torch.from_numpy(dataset[ind]['WC'])
            x_c = torch.from_numpy(dataset[ind]['x_c'])
            WC[b] = wc
            X_c[b] = x_c
        
        WW = Variable(WW, volatile=volatile)
        X = Variable(X, volatile=volatile)
        WC = Variable(WC, volatile=volatile)
        X_c = Variable(X_c, volatile=volatile)

        if cuda:
            return [WW.cuda(), X.cuda()], [WC.cuda(), X_c.cuda()]
        else:
            return [WW, X], [WC, X_c]

if __name__ == '__main__':
    # Test Generator module
    N = 40
    path_dataset = '/data/anowak/TSP/'
    path_tsp = '/home/anowak/QAP_pt/src/tsp/LKH/'
    gen = Generator(path_dataset, path_tsp)
    gen.num_examples_train = 20000
    gen.num_examples_test = 4000
    gen.N = N
    gen.load_dataset()
    # g1, g2 = gen.sample_batch(32, cuda=False)
    # print(g1[0].size())
    # print(g1[0][0].data.cpu().numpy())
    print('Dataset of length {} created.'.format(N))

