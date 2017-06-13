import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from data_generator import Generator
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)

class Logger(object):
    def __init__(self, path_logger):
        directory = os.path.join(path_logger, 'plots/')
        self.path = path_logger
        self.path_dir = directory
        # Create directory if necessary
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        self.loss_train = []
        self.loss_test = []
        self.accuracy_train = []
        self.accuracy_test = []
        self.args = None

    def write_settings(self, args):
        self.args = {}
        # write info
        path = os.path.join(self.path, 'experiment.txt')
        with open(path, 'w') as file:
            for arg in vars(args):
                file.write(str(arg) + ' : ' + str(getattr(args, arg)) + '\n')
                self.args[str(arg)] = getattr(args, arg)

    def add_train_loss(self, loss):
        self.loss_train.append(loss.data.cpu().numpy())

    def add_test_loss(self, loss):
        self.loss_test.append(loss)

    def add_train_accuracy(self, pred, labels, W):
        # accuracy = utils.compute_recovery_rate(pred, labels)
        # accuracy = utils.compute_accuracy(pred, labels)
        accuracy = utils.compute_mean_cost(pred, W)
        self.accuracy_train.append(accuracy)
        if len(self.accuracy_train) > 20:
           self.accuracy_train[-1] = sum(self.accuracy_train[-20:])/20.0

    def add_test_accuracy(self, pred, labels, W):
        # accuracy = utils.compute_recovery_rate(pred, labels)
        # accuracy = utils.compute_accuracy(pred, labels)
        accuracy = utils.compute_mean_cost(pred, W)
        self.accuracy_test.append(accuracy)

    def plot_train_loss(self):
        plt.figure(0)
        plt.clf()
        iters = range(len(self.loss_train))
        plt.semilogy(iters, self.loss_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training Loss: p={}, p_e={}'
                  .format(self.args['edge_density'], self.args['noise']))
        path = os.path.join(self.path_dir, 'training_loss.png') 
        plt.savefig(path)

    def plot_test_loss(self):
        plt.figure(1)
        plt.clf()
        test_freq = self.args['test_freq']
        iters = test_freq * range(len(self.loss_test))
        plt.semilogy(iters, self.loss_test, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Testing Loss: p={}, p_e={}'
                  .format(self.args['edge_density'], self.args['noise']))
        path = os.path.join(self.path_dir, 'testing_loss.png') 
        plt.savefig(path)

    def plot_train_accuracy(self):
        plt.figure(0)
        plt.clf()
        iters = range(len(self.accuracy_train))
        plt.plot(iters, self.accuracy_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy: p={}, p_e={}'
                  .format(self.args['edge_density'], self.args['noise']))
        path = os.path.join(self.path_dir, 'training_accuracy.png') 
        plt.savefig(path)

    def plot_test_accuracy(self):
        plt.figure(1)
        plt.clf()
        test_freq = self.args['test_freq']
        iters = test_freq * range(len(self.accuracy_test))
        plt.plot(iters, self.accuracy_test, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Accuracy')
        plt.title('Testing Accuracy: p={}, p_e={}'
                  .format(self.args['edge_density'], self.args['noise']))
        path = os.path.join(self.path_dir, 'testing_accuracy.png') 
        plt.savefig(path)

if __name__ == '__main__':
    path_dataset = '/data/anowak/TSP/'
    path_logger = '/home/anowak/tmp/TSP1/'
    path_tsp = '/home/anowak/QAP_pt/src/tsp/LKH/'
    gen = Generator(path_dataset, path_tsp, mode='CEIL_2D')
    gen.num_examples_train = 200
    gen.num_examples_test = 40
    gen.J = 4
    gen.load_dataset()
    sample = gen.sample_batch(1, cuda=torch.cuda.is_available())
    W = sample[0][0][:, :, :, 1]
    pred = sample[1][0]
    perm = sample[1][1]
    optimal_costs = sample[2]
    costs, = utils.compute_hamcycle(pred, W)
    print('W', W)
    print('oracle perm', perm)
    print('costs', costs[0])
    print('optimal_costs', optimal_costs)
    print(costs[0]/optimal_costs)
