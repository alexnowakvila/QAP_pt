import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.rcParams.update({'font.size': 22})
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
        self.loss_test_aux = []
        self.accuracy_train = []
        self.accuracy_test = []
        self.accuracy_test_aux = []
        self.cost_train = []
        self.cost_test = []
        self.cost_test_oracle = []
        self.cost_test_aux = []
        self.cost_test_aux_oracle = []
        self.path_examples = None
        self.args = None

    def write_settings(self, args):
        self.args = {}
        # write info
        path = os.path.join(self.path, 'experiment.txt')
        with open(path, 'w') as file:
            for arg in vars(args):
                file.write(str(arg) + ' : ' + str(getattr(args, arg)) + '\n')
                self.args[str(arg)] = getattr(args, arg)

    def save_model(self, model):
        save_dir = os.path.join(self.path, 'parameters/')
        # Create directory if necessary
        try:
            os.stat(save_dir)
        except:
            os.mkdir(save_dir)
        path = os.path.join(save_dir, 'gnn.pt')
        torch.save(model, path)
        print('Model Saved.')

    def load_model(self, parameters_path):
        path = os.path.join(parameters_path, 'parameters/gnn.pt')
        if os.path.exists(path):
            print('GNN successfully loaded from {}'.format(path))
            return torch.load(path)
        else:
            raise ValueError('Parameter path {} does not exist.'
                             .format(path))


    def plot_example(self, Paths, costs, oracle_costs, Perms,
                     Cities, num_plots=1):
        num_plots = min(num_plots, Paths.size(0))
        num_plots = 1
        for fig in range(num_plots):
            cost = costs[fig]
            oracle_cost = oracle_costs[fig]
            predicted_path = Paths[fig].cpu().numpy()
            oracle_path = Perms[fig].cpu().numpy()
            cities = Cities[fig].cpu().numpy()
            oracle_path = oracle_path.astype(int)
            # print('predicted path: ', predicted_path)
            # print('oracle path: ', oracle_path)
            oracle_cities = cities[oracle_path]
            predicted_cities = cities[predicted_path]
            oracle_cities = (np.concatenate((oracle_cities, np.expand_dims(
                             oracle_cities[0], axis=0)), axis=0))
            predicted_cities = (np.concatenate((predicted_cities, np.
                                expand_dims(predicted_cities[0], axis=0)),
                                axis=0))
            plt.figure(2, figsize=(12, 12))
            plt.clf()
            plt.scatter(cities[:, 0], cities[:, 1], c='b')
            plt.plot(oracle_cities[:, 0], oracle_cities[:, 1], c='r')
            plt.title('Target: {0:.3f}'
                      .format(20*np.sqrt(2)-oracle_cost), fontsize=100)
            path = os.path.join(self.path_dir, 'ground_tsp{}.eps'.format(fig))
            plt.savefig(path, format='eps')

            plt.figure(2, figsize=(12, 12))
            plt.clf()
            plt.scatter(cities[:, 0], cities[:, 1], c='b')
            plt.plot(predicted_cities[:, 0], predicted_cities[:, 1], c='g')
            plt.title('Predicted: {0:.3f}'
                      .format(20*np.sqrt(2) - cost), fontsize=100)
            path = os.path.join(self.path_dir, 'pred_tsp{}.eps'.format(fig))
            plt.savefig(path, format='eps')

    def add_train_loss(self, loss):
        self.loss_train.append(loss.data.cpu().numpy())

    def add_test_loss(self, loss, last=False):
        self.loss_test_aux.append(loss.data.cpu().numpy())
        if last:
            loss_test = np.array(self.loss_test_aux).mean()
            self.loss_test.append(loss_test)
            self.loss_test_aux = []

    def add_train_accuracy(self, pred, labels, W):
        accuracy = utils.compute_accuracy(pred, labels)
        costs = utils.compute_mean_cost(pred, W)
        self.accuracy_train.append(accuracy)
        self.cost_train.append(sum(costs) / float(len(costs)))

    def add_test_accuracy(self, pred, labels, perms, W, cities, oracle_costs,
                          last=False, beam_size=2):
        accuracy = utils.compute_accuracy(pred, labels)
        costs, Paths = utils.beamsearch_hamcycle(pred.data, W.data,
                                                 beam_size=beam_size)
        self.accuracy_test_aux.append(accuracy)
        self.cost_test_aux.append(np.array(costs.cpu().numpy()).mean())
        self.cost_test_aux_oracle.append(np.array(oracle_costs).mean())
        if last:
            accuracy_test = np.array(self.accuracy_test_aux).mean()
            self.accuracy_test.append(accuracy_test)
            self.accuracy_test_aux = []
            cost_test = np.array(self.cost_test_aux).mean()
            self.cost_test.append(cost_test)
            self.cost_test_aux = []
            cost_test_oracle = np.array(self.cost_test_aux_oracle).mean()
            self.cost_test_oracle.append(cost_test_oracle)
            self.cost_test_aux_oracle = []
            self.plot_example(Paths, costs, oracle_costs, perms, cities)

    def plot_train_logs(self):
        plt.figure(0, figsize=(20, 20))
        plt.clf()
        # plot loss
        plt.subplot(3, 1, 1)
        iters = range(len(self.loss_train))
        plt.semilogy(iters, self.loss_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training Loss')
        # plot accuracy
        plt.subplot(3, 1, 2)
        iters = range(len(self.accuracy_train))
        plt.plot(iters, self.accuracy_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        # plot costs
        plt.subplot(3, 1, 3)
        iters = range(len(self.cost_train))
        plt.plot(iters, self.cost_train, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Average Mean cost')
        plt.title('Average Mean cost Training')
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
        path = os.path.join(self.path_dir, 'training.png') 
        plt.savefig(path)

    def plot_test_logs(self):
        plt.figure(1, figsize=(20, 20))
        plt.clf()
        # plot loss
        plt.subplot(3, 1, 1)
        test_freq = self.args['test_freq']
        iters = test_freq * np.arange(len(self.loss_test))
        plt.semilogy(iters, self.loss_test, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Testing Loss')
        # plot accuracy
        plt.subplot(3, 1, 2)
        iters = test_freq * np.arange(len(self.accuracy_test))
        plt.plot(iters, self.accuracy_test, 'b')
        plt.xlabel('iterations')
        plt.ylabel('Accuracy')
        plt.title('Testing Accuracy')
        # plot costs
        plt.subplot(3, 1, 3)
        beam_size = self.args['beam_size']
        iters = range(len(self.cost_test))
        plt.plot(iters, self.cost_test, 'b')
        print('COST ORACLE', self.cost_test_oracle[-1])
        plt.plot(iters, self.cost_test_oracle, 'r')
        plt.xlabel('iterations')
        plt.ylabel('Mean cost')
        plt.title('Mean cost Testing with beam_size : {}'.format(beam_size))
        plt.tight_layout(pad=0.8, w_pad=0.5, h_pad=2.0)
        path = os.path.join(self.path_dir, 'testing.png') 
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
    sample = gen.sample_batch(2, cuda=torch.cuda.is_available())
    W = sample[0][0][:, :, :, 1] # weighted adjacency matrix
    WTSP = sample[1][0] # hamiltonian cycle adjacency matrix
    pred = sample[1][0]
    perm = sample[1][1]
    optimal_costs = sample[2]
    ########################## test compute accuracy ##########################
    labels = torch.topk(WTSP, 2, dim=2)[1]
    accuracy = utils.compute_accuracy(WTSP, labels)
    print('accuracy', accuracy)
    ########################## test compute_hamcycle ##########################
    # costs, = utils.greedy_hamcycle(pred, W)
    # print('W', W)
    # print('oracle perm', perm)
    # print('costs', costs[0])
    # print('optimal_costs', optimal_costs)
    # print(costs[0]/optimal_costs)
    ############################# test beamsearch #############################
    # costs, paths = utils.beamsearch_hamcycle(WTSP.data, W.data)
    # print('paths', paths)
    # print('WTSP', WTSP)
    # print('W', W)
