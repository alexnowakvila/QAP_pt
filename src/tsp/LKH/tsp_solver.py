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

parser = argparse.ArgumentParser()

###############################################################################
#                          TSP SOLVER settings                                #
###############################################################################

parser.add_argument('--ASCENT_CANDIDATES', nargs='?', const=1, type=int,
                    default=50, help='The number of candidate edges to be'
                    'associated with each node during the ascent. The '
                    'candidate set is complemented such that every '
                    'candidate edge is associated with both its two end'
                    'nodes.')
parser.add_argument('--BACKBONE_TRIALS', nargs='?', const=1, type=int,
                    default=50, help='The number of backbone trials in each'
                    'run.')
parser.add_argument('--BACKTRACKING', nargs='?', const=1, type=str,
                    default='NO', help='Specifies whether a backtracking K-opt'
                    'move is to be used as the first move in a sequence of'
                    'moves (where K = MOVE_TYPE).')
parser.add_argument('--CANDIDATE_FILE', nargs='?', const=1, type=str,
                    help='Specifies the name of a file to which the candidate'
                    'sets are to be written. If, however, the file already'
                    'exists, the candidate edges are read from the file. '
                    'The first line of the file contains the dimension of'
                    'the instance. Each of the following lines contains a node'
                    'number, the number of the dad of the node'
                    'in the minimum spanning tree (0, if the node has no dad),'
                    ' the number of candidate edges emanating from the node,'
                    'followed by the candidate edges. For each candidate edge'
                    'its end node number and alpha-value are given. It is'
                    'possible to give more than one CANDIDATE_FILE '
                    'specification. In this case the given files are read and'
                    'the union of their candidate edges is used as candidate'
                    'sets.')
parser.add_argument('--CANDIDATE_SET_TYPE', nargs='?', const=1, type=str,
                    default='ALPHA', help='{ ALPHA | DELAUNAY [ PURE ] |'
                    'NEAREST-NEIGHBOR | QUADRANT }.'
                    'Specifies the candidate set type. ALPHA is LKHs default '
                    'type. It is applicable in general. The other'
                    'three types can only be used for instances given by '
                    'coordinates. The optional suffix PURE for the'
                    'DELAUNAY type specifies that only edges of the Delaunay'
                    'graph are used as candidates.')
# TO DO: Finish list
args = parser.parse_args()

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
        self.header_template_par = ('PROBLEM_FILE = {}\nMOVE_TYPE = {:d}\n'
                                    'PATCHING_C = {:d}\nPATCHING_A = {:d}\n'
                                    'RUNS = {:d}\nTOUR_FILE = {}\n')
        self.header_template_tsp = ('NAME : {}\nCOMMENT : {}\nTYPE : TSP \n'
                                    'DIMENSION : {}\nEDGE_WEIGHT_TYPE : {}\n')
        self.dataset = {'MAPS': [], 'HAM_CYCLES': [], 'LENGTH_TOURS': []}

    def cities_generator(self, N):
        cities = np.random.uniform(0, 1, [N, 2])
        return cities

    def save_solverformat(self, example, id, mode='EUC_2D'):
        if mode == 'EUC_2D':
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
                HEADER = self.header_template_par.format(*HEADER)
                file.write(HEADER)
            # create .tsp file
            with open(path_tsp, 'w+') as file:
                HEADER = ['pr{}.tsp'.format(id),
                          '{}-city problem'.format(example.shape[0]),
                          example.shape[0], mode]
                HEADER = self.header_template_tsp.format(*HEADER)
                file.write(HEADER)
                file.write('NODE_COORD_SECTION \n')
                example_int = (example * self.C).astype(int)
                for i in range(example.shape[0]):
                    node = ('{:<2} {:<8e} {:<8e} \n'
                            .format(i + 1, *list(example_int[i])))
                    file.write(node)
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
            ham_cycle.append(int(content[length + 6][:-1]))
        return ham_cycle, length_tour

    def plot_example(self, x, path):
        MAP, HAM_CYCLE, LENGTH_TOUR = x
        perm = np.array(HAM_CYCLE[:-1]) - 1
        MAP = MAP[perm]
        MAP = np.concatenate((MAP, np.expand_dims(MAP[0], axis=0)), axis=0)
        plt.figure(0)
        plt.clf()
        plt.scatter(MAP[:, 0], MAP[:, 1], c='b')
        plt.plot(MAP[:, 0], MAP[:, 1], c='r')
        plt.savefig(path)

    def create_dataset(self, num_examples, N, mode='EUC_2D'):
        MAPS = []
        HAM_CYCLES = []
        LENGTH_TOURS = []
        for example in range(num_examples):
            MAP = self.cities_generator(N)
            self.save_solverformat(MAP, 0, mode=mode)
            self.tsp_solver(0)
            HAM_CYCLE, LENGTH_TOUR = self.extract_path(0)
            MAPS.append(MAP)
            HAM_CYCLES.append(HAM_CYCLE)
            LENGTH_TOURS.append(LENGTH_TOUR / self.C)
            print('example {} created'.format(example))
            print(HAM_CYCLE)
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
    num_examples = 20
    N = 100
    gen.create_dataset(num_examples, N)
    map10 = gen.dataset['MAPS'][0]
    length_tour10 = gen.dataset['LENGTH_TOURS'][0]
    cycle10 = gen.dataset['HAM_CYCLES'][0]
    # print('map', map10)
    # print('len_tour', length_tour10)
    # print('cycle10', cycle10)
    x = map10, cycle10, length_tour10
    gen.plot_example(x,  plot_path)

