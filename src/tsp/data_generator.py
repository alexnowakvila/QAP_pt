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
import math

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class Generator(TSP):
    def __init__(self, path_dataset, path_tsp, mode='CEIL_2D'):
        super().__init__(path_tsp)
        # TSP.__init__(self, path_dataset)
        self.path_dataset = path_dataset
        self.num_examples_train = 10e6
        self.num_examples_test = 10e4
        self.data_train = []
        self.data_test = []
        self.N = 20
        self.J = 4
        self.mode = mode
        self.sym = True

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
            return math.ceil(np.sqrt(np.square(x - y).sum()))
        def l1_dist(x, y):
            return np.abs(x - y).sum()
        for i in range(0, N - 1):
            for j in range(i + 1, N):
                city1 = cities[i]*self.C
                city2 = cities[j]*self.C
                dist = l2_dist(city1, city2)/float(self.C)
                W[i, j] = np.sqrt(2) - float(dist)
                W[j, i] = W[i, j]
        return W

    def cycle_adj(self, N, sym=False):
        W = np.zeros((N, N))
        if sym:
            W[N-1, N-2] = 1
            W[N-1, 0] = 1
            W[0, 1] = 1
            W[0, N-1] = 1
            for i in range(1, N-1):
                W[i, i-1] = 1
                W[i, i+1] = 1
        else:
            W[N-1, N-2] = 0
            W[N-1, 0] = 0
            W[0, 1] = 1
            W[0, N-1] = 1
            for i in range(1, N-1):
                W[i, i-1] = 0
                W[i, i+1] = 1
        return W

    def compute_example(self):
        example = {}
        if self.mode == 'CEIL_2D':
            cities = self.cities_generator(self.N)
            W = self.adj_from_coord(cities)
            WW, x = self.compute_operators(W)
            # add_coordinates
            x = np.concatenate([x, cities], axis=1)
            example['cities'] = cities
            example['WW'], example['x'] = WW, x
            # compute hamiltonian cycle
            self.save_solverformat(cities, self.N, mode='CEIL_2D')
        elif self.mode == 'EXPLICIT':
            W = self.adj_generator(self.N)
            WW, x = self.compute_operators(W)
            example['WW'], example['x'] = WW, x
            # compute hamiltonian cycle
            self.save_solverformat(W, self.N, mode='EXPLICIT')
            raise ValueError('Mode {} not yet supported.'.format(mode))
        else:
            raise ValueError('Mode {} not supported.'.format(mode))
        self.tsp_solver(self.N)
        # print(cities)
        ham_cycle, length_cycle = self.extract_path(self.N)
        example['HAM_cycle'] = ham_cycle
        cost = float(length_cycle)/float(self.C)
        example['Length_cycle'] = np.sqrt(2)*self.N - cost
        example['WTSP'] = self.perm_to_adj(ham_cycle, self.N)
        example['labels'] = self.perm_to_labels(ham_cycle, self.N,
                                                sym=self.sym)
        example['perm'] = ham_cycle
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
        filename = 'TSP{}{}train.np'.format(self.N, self.mode)
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
        filename = 'TSP{}{}test.np'.format(self.N, self.mode)
        path = os.path.join(self.path_dataset, filename)
        if os.path.exists(path):
            print('Reading testing dataset at {}'.format(path))
            self.data_test = np.load(open(path, 'rb'))
        else:
            print('Creating testing dataset.')
            self.create_dataset_test()
            print('Saving testing datatset at {}'.format(path))
            np.save(open(path, 'wb'), self.data_test)

    def sample_batch(self, num_samples, is_training=True, it=0,
                     cuda=True, volatile=False):
        WW_size = self.data_train[0]['WW'].shape
        x_size = self.data_train[0]['x'].shape

        # define batch elements
        WW = torch.zeros(num_samples, *WW_size)
        X = torch.zeros(num_samples, *x_size)
        WTSP = torch.zeros(num_samples, *WW_size[:-1])
        if self.sym:
            P = torch.zeros(num_samples, self.N, 2)
        else:
            P = torch.zeros(num_samples, self.N)
        Cities = torch.zeros((num_samples, WW_size[1], 2))
        Perm = torch.zeros((num_samples, WW_size[1]))
        Cost = np.zeros(num_samples)
        # fill batch elements 
        if is_training:
            dataset = self.data_train
        else:
            dataset = self.data_test
        for b in range(num_samples):
            if is_training:
                # random element in the dataset
                ind = np.random.randint(0, len(dataset))
            else:
                ind = it * num_samples + b
            ww = torch.from_numpy(dataset[ind]['WW'])
            x = torch.from_numpy(dataset[ind]['x'])
            WW[b], X[b] = ww, x
            WTSP[b] = torch.from_numpy(dataset[ind]['WTSP'])
            P[b] = torch.from_numpy(dataset[ind]['labels'])
            Cities[b] = torch.from_numpy(dataset[ind]['cities'])
            Perm[b] = torch.from_numpy(dataset[ind]['perm'])
            Cost[b] = dataset[ind]['Length_cycle']
        # wrap as variables
        WW = Variable(WW, volatile=volatile)
        X = Variable(X, volatile=volatile)
        WTSP = Variable(WTSP, volatile=volatile)
        P = Variable(P, volatile=volatile)
        if cuda:
            return ([WW.cuda(), X.cuda()], [WTSP.cuda(), P.cuda()],
                    Cities.cuda(), Perm.cuda(), Cost)
        else:
            return [WW, X], [WTSP, P], Cities, Perm, Cost

if __name__ == '__main__':
    # Test Generator module
    N = 50
    path_dataset = '/data/anowak/TSP/'
    path_tsp = '/home/anowak/QAP_pt/src/tsp/LKH/'
    gen = Generator(path_dataset, path_tsp)
    gen.num_examples_train = 20000
    gen.num_examples_test = 1000
    gen.N = N
    gen.load_dataset()
    out = gen.sample_batch(32, cuda=False)
    # print(g1[0].size())
    # print(g1[0][0].data.cpu().numpy())
    print('Dataset of length {} created.'.format(N))

