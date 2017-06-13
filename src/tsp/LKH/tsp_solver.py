#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
from subprocess import Popen, PIPE, STDOUT, call
import os
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#Pytorch requirements
import unicodedata
import string
import re
import random
import argparse

class TSP(object):
    def __init__(self, path_tsp):
        self.C = 10e4
        self.path_solver = path_tsp
        self.path_datatsp = path_tsp + 'DATA/'
        # export LKH_PATH
        cmd = "export LKH_PATH='{}'".format(self.path_solver)
        p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                  close_fds=True)
        # print('out1', p.stdout.read())
        cmd = "export PATH=$PATH:$LKH_PATH"
        p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                  close_fds=True)
        # print('out2', p.stdout.read())
        self.header_temp_par = ('PROBLEM_FILE = {}\nMOVE_TYPE = {:d}\n'
                                    'PATCHING_C = {:d}\nPATCHING_A = {:d}\n'
                                    'RUNS = {:d}\nTOUR_FILE = {}\n')
        self.header_temp_euc2dtsp = ('NAME : {}\nCOMMENT : {}\nTYPE : TSP \n'
                                     'DIMENSION : {}\nEDGE_WEIGHT_TYPE : {}\n')
        self.header_temp_graphtsp = ('NAME : {}\nCOMMENT : {}\nTYPE : TSP \n'
                                     'DIMENSION : {}\nEDGE_WEIGHT_TYPE : {}\n'
                                     'EDGE_WEIGHT_FORMAT : FULL_MATRIX\n')
        self.dataset = {'MAPS': [], 'HAM_CYCLES': [], 'LENGTH_TOURS': []}

    def cities_generator(self, N):
        cities = np.random.uniform(0, 1, [N, 2])
        return cities

    def adj_generator(self, N):
        return np.random.uniform(0, 1, [N, N])

    def perm_to_adj(self, perm, N):
        W = np.zeros((N, N))
        perm = list(perm[1:]) + [perm[0]]
        W[perm[0], perm[1]] = 1
        W[perm[0], perm[N-1]] = 1
        W[perm[N-1], perm[N-2]] = 1
        W[perm[N-1], perm[0]] = 1
        for i in range(1, N-1):
            W[perm[i], perm[i-1]] = 1
            W[perm[i], perm[i+1]] = 1
        return W

    def perm_to_labels(self, perm, N, sym=True):
        if sym:
            labels = np.zeros((N, 2))
            labels[perm[0], 0] = perm[1]
            labels[perm[0], 1] = perm[N-1]
            labels[perm[N-1], 0] = perm[0]
            labels[perm[N-1], 1] = perm[N-2]
            for i in range(1, N-1):
                labels[perm[i], 0] = perm[i+1]
                labels[perm[i], 1] = perm[i-1]
        else:
            labels = np.zeros(N)
            labels[perm[N-1]] = perm[0]
            for i in range(N-1):
                labels[perm[i]] = perm[i+1]
        return labels

    def save_solverformat(self, example, id, mode='CEIL_2D'):
        if mode == 'CEIL_2D':
            path_par = os.path.join(self.path_datatsp, 'pr{}.par'.format(id))
            path_tsp = os.path.join(self.path_datatsp, 'pr{}.tsp'.format(id))
            path_res = os.path.join(self.path_datatsp, 'res{}.tsp'.format(id))
            # create .par file
            with open(path_par, 'w+') as file:
                MOVE_TYPE = 5
                PATCHING_C = 3
                PATCHING_A = 2
                RUNS = 10
                TOUR_FILE = path_res
                HEADER = [path_tsp,
                          MOVE_TYPE,
                          PATCHING_C,
                          PATCHING_A,
                          RUNS,
                          TOUR_FILE]
                HEADER = self.header_temp_par.format(*HEADER)
                file.write(HEADER)
            # create .tsp file
            with open(path_tsp, 'w+') as file:
                HEADER = ['pr{}.tsp'.format(id),
                          '{}-city problem'.format(example.shape[0]),
                          example.shape[0], mode]
                HEADER = self.header_temp_euc2dtsp.format(*HEADER)
                file.write(HEADER)
                file.write('NODE_COORD_SECTION \n')
                example_int = (example * self.C).astype(int)
                for i in range(example.shape[0]):
                    node = ('{:<2} {:<8e} {:<8e} \n'
                            .format(i + 1, *list(example_int[i])))
                    file.write(node)
                file.write('EOF \n')
        elif mode == 'EXPLICIT':
            path_par = os.path.join(self.path_datatsp, 'pr{}.par'.format(id))
            path_tsp = os.path.join(self.path_datatsp, 'pr{}.tsp'.format(id))
            path_res = os.path.join(self.path_datatsp, 'res{}.tsp'.format(id))
            # create .par file
            with open(path_par, 'w+') as file:
                MOVE_TYPE = 5
                PATCHING_C = 3
                PATCHING_A = 2
                RUNS = 10
                TOUR_FILE = path_res
                HEADER = [path_tsp,
                          MOVE_TYPE,
                          PATCHING_C,
                          PATCHING_A,
                          RUNS,
                          TOUR_FILE]
                HEADER = self.header_temp_par.format(*HEADER)
                file.write(HEADER)
            # create .tsp file
            with open(path_tsp, 'w+') as file:
                HEADER = ['pr{}.tsp'.format(id),
                          '{}-graph problem'.format(example.shape[0]),
                          example.shape[0], mode]
                HEADER = self.header_temp_graphtsp.format(*HEADER)
                file.write(HEADER)
                file.write('EDGE_WEIGHT_SECTION \n')
                example_int = (example * self.C).astype(int)
                for i in range(example.shape[0]):
                    row = []
                    for j in range(example.shape[1]):
                        row.append(example_int[i, j])
                    row = ' '.join('{:<8d}'.format(edge) for edge in row)
                    file.write(row + '\n')
                file.write('EOF \n')
        else:
            raise ValueError('TSP mode {} not supported.'.format(mode))

    def tsp_solver(self, id):
        path_exec = os.path.join(self.path_solver, 'LKH')
        path_example = os.path.join(self.path_datatsp, 'pr{}.par'.format(id))
        cmd = "{} {}".format(path_exec, path_example)
        p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                  close_fds=True)
        output = p.stdout.read()
        return output

    def extract_path(self, id):
        ham_cycle = []
        length_tour = 0
        path_res = os.path.join(self.path_datatsp, 'res{}.tsp'.format(id))
        with open(path_res, 'r') as file:
            content = file.readlines()
            tour = content[1]
            length = int(content[4][11:-1])
            length_tour = int(tour[19:-1])
            for node in range(length):
                ham_cycle.append(int(content[node + 6][:-1]))
        # print(ham_cycle)
        ham_cycle = np.array(ham_cycle) - 1
        return ham_cycle, length_tour

    def plot_example(self, x, path, mode='CEIL_2D'):
        MAP, HAM_CYCLE, LENGTH_TOUR = x
        perm = np.array(HAM_CYCLE)
        if mode == 'CEIL_2D':
            MAP = MAP[perm]
            MAP = np.concatenate((MAP, np.expand_dims(MAP[0], axis=0)), axis=0)
            plt.figure(0)
            plt.clf()
            plt.scatter(MAP[:, 0], MAP[:, 1], c='b')
            plt.plot(MAP[:, 0], MAP[:, 1], c='r')
            plt.savefig(path)
        elif mode == 'EXPLICIT':
            pass
        else:
            raise ValueError('Plot for Mode {} not supported.'.format(mode))

    def create_dataset(self, num_examples, N, mode='CEIL_2D'):
        MAPS = []
        HAM_CYCLES = []
        LENGTH_TOURS = []
        for example in range(num_examples):
            if mode == 'CEIL_2D':
                MAP = self.cities_generator(N)
            elif mode == 'EXPLICIT':
                MAP = self.adj_generator(N)
            self.save_solverformat(MAP, 0, mode=mode)
            self.tsp_solver(0)
            HAM_CYCLE, LENGTH_TOUR = self.extract_path(0)
            MAPS.append(MAP)
            HAM_CYCLES.append(HAM_CYCLE)
            LENGTH_TOURS.append(LENGTH_TOUR / self.C)
            print('example {} created'.format(example))
            # print(HAM_CYCLE)
        self.dataset['MAPS'] = MAPS
        self.dataset['HAM_CYCLES'] = HAM_CYCLES
        self.dataset['LENGTH_TOURS'] = LENGTH_TOURS

if __name__ == '__main__':
    path_tsp = '/home/anowak/QAP_pt/src/tsp/LKH/'
    plot_path = '/home/anowak/QAP_pt/plots/tsp.png'
    gen = TSP(path_tsp)
    # example = gen.cities_generator(50)
    # gen.save_solverformat(example, 0)
    # out = gen.tsp_solver(0)
    # ham_cycle, lt = gen.extract_path(0)
    num_examples = 5
    N = 10
    gen.create_dataset(num_examples, N, mode='CEIL_2D')
    map10 = gen.dataset['MAPS'][0]
    length_tour10 = gen.dataset['LENGTH_TOURS'][0]
    cycle10 = gen.dataset['HAM_CYCLES'][0]
    # print('map', map10)
    # print('len_tour', length_tour10)
    # print('cycle10', cycle10)
    x = map10, cycle10, length_tour10
    gen.plot_example(x,  plot_path)
    print('cycle', cycle10)
    print('W', gen.perm_to_adj(cycle10, N))
    print('labels', gen.perm_to_labels(cycle10, N, sym=True))
    print('len_tour', length_tour10)

