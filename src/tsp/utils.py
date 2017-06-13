#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
# import dependencies
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from beam_search import BeamSearch
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

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    torch.manual_seed(0)

def compute_accuracy(pred, labels):
    pred = torch.topk(pred, 2, dim=2)[1]
    p = torch.sort(pred, 2)[0]
    l = torch.sort(labels, 2)[0]
    # print('pred', p)
    # print('labels', l)
    error = 1 - torch.eq(p, l).min(2)[0].type(dtype).squeeze(2)
    frob_norm = error.mean(1).squeeze(1)
    accuracy = 1 - frob_norm
    accuracy = accuracy.mean(0).squeeze()
    return accuracy.data.cpu().numpy()[0]

def compute_mean_cost(pred, W):
    # cost estimator for training time
    mean_rowcost = torch.mul(pred, W).mean(2).squeeze(2)
    return mean_rowcost.mean(1).mean(0).squeeze()

def compute_recovery_rate(pred, labels):
    pred = pred.max(2)[1]
    error = 1 - torch.eq(pred, labels).type(dtype).squeeze(2)
    frob_norm = error.mean(1).squeeze(1)
    accuracy = 1 - frob_norm
    accuracy = accuracy.mean(0).squeeze()
    return accuracy.data.cpu().numpy()[0]

def compute_hamcycle(pred, W):
    def next_vertex(start, prev, pred):
        nxt = pred[start].data.cpu().numpy()
        col = int(nxt[0] == prev)
        end = nxt[col]
        return end
    N = W.size(-1)
    batch_size = W.size(0)
    Costs = []
    Paths = []
    pred = torch.topk(pred, 2, dim=2)[1]
    for b in range(batch_size):
        cost = 0.0
        path = [0]
        predb = pred[b]
        Wb = W[b]
        start = 0
        end = next_vertex(start, -1, predb)
        # print(start, end)
        for i in range(N-1):
            cost += Wb[start, end]
            path.append(end)
            prev = start
            start = end
            end = next_vertex(start, prev, predb)
            # print(start, end)
        cost += Wb[start, end]
        Costs.append(cost.data.cpu().numpy())
        Paths.append(path)
    return Costs, Paths

def compute_cost_path(Paths, W):
    # Paths is a list of length N+1
    batch_size = W.size(0)
    N = W.size(-1)
    Costs = []
    for b in range(batch_size):
        path = Paths[b].squeeze(0)
        Wb = W[b].squeeze(0)
        cost = 0.0
        for node in range(N-1):
            start = path[node]
            end = path[node + 1]
            cost += Wb[start, end]
        cost += Wb[end, 0]
        Costs.append(cost)
    return Costs

def beamsearch_hamcycle(pred, W, beam_size=2):
    N = W.size(-1)
    batch_size = W.size(0)
    BS = BeamSearch(beam_size, batch_size, N)
    trans_probs = pred.gather(1, BS.get_current_state())
    for step in range(N-1):
        BS.advance(trans_probs, step + 1)
        trans_probs = pred.gather(1, BS.get_current_state())
    ends = torch.zeros(batch_size, 1).type(dtype_l)
    # extract paths
    Paths = BS.get_hyp(ends)
    # Compute cost of path
    Costs = compute_cost_path(Paths, W)
    return Costs, Paths