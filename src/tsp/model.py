#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib
matplotlib.use('Agg')

# Pytorch requirements
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.cuda.LongTensor

def sinkhorn_knopp(A, iterations=1):
    A_size = A.size()
    for it in range(iterations):
        A = A.view(A_size[0]*A_size[1], A_size[2])
        A = F.softmax(A)
        A = A.view(*A_size).permute(0, 2, 1)
        A = A.view(A_size[0]*A_size[1], A_size[2])
        A = F.softmax(A)
        A = A.view(*A_size).permute(0, 2, 1)
    return A

def gmul(input):
    W, x = input
    # x is a tensor of size (bs, N, num_features)
    # W is a tensor of size (bs, N, N, J)
    x_size = x.size()
    W_size = W.size()
    N = W_size[-2]
    J = W_size[-1]
    W = W.split(1, 3)
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x) # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output

class Gconv_last(nn.Module):
    def __init__(self, feature_maps, J):
        super(Gconv_last, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_outputs = feature_maps[2]
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, input):
        W = input[0]
        x = gmul(input) # out has size (bs, N, num_inputs)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(x_size[0]*x_size[1], -1)
        x = self.fc(x) # has size (bs*N, num_outputs)
        x = x.view(*x_size[:-1], self.num_outputs)
        return W, x

class Gconv(nn.Module):
    def __init__(self, feature_maps, J):
        super(Gconv, self).__init__()
        self.num_inputs = J*feature_maps[0]
        self.num_outputs = feature_maps[2]
        self.fc1 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.fc2 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, input):
        W = input[0]
        x = gmul(input) # out has size (bs, N, num_inputs)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        x1 = F.relu(self.fc1(x)) # has size (bs*N, num_outputs)
        x2 = self.fc2(x)
        x = torch.cat((x1, x2), 1)
        x = self.bn(x)
        x = x.view(*x_size[:-1], self.num_outputs)
        return W, x

class GNN(nn.Module):
    def __init__(self, num_features, num_layers, J):
        super(GNN, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.featuremap_in = [1, 1, num_features]
        self.featuremap_mi = [num_features, num_features, num_features]
        self.featuremap_end = [num_features, num_features, num_features]
        self.layer0 = Gconv(self.featuremap_in, J)
        for i in range(num_layers):
            module = Gconv(self.featuremap_mi, J)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = Gconv_last(self.featuremap_end, J)

    def forward(self, input):
        cur = self.layer0(input)
        for i in range(self.num_layers):
            cur = self._modules['layer{}'.format(i+1)](cur)
        out = self.layerlast(cur)
        return out[1]

class Siamese_GNN(nn.Module):
    def __init__(self, num_features, num_layers, J):
        super(Siamese_GNN, self).__init__()
        self.gnn = GNN(num_features, num_layers, J)

    def forward(self, g1, g2):
        emb1 = self.gnn(g1)
        emb2 = self.gnn(g2)
        # embx are tensors of size (bs, N, num_features)
        out = torch.bmm(emb1, emb2.permute(0, 2, 1))
        return out # out has size (bs, N, N)

if __name__ == '__main__':
    # test modules
    bs =  4
    num_features = 10
    num_layers = 5
    N = 8
    x = torch.ones((bs, N, num_features))
    W1 = torch.eye(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    W2 = torch.ones(N).unsqueeze(0).unsqueeze(-1).expand(bs, N, N, 1)
    J = 2
    W = torch.cat((W1, W2), 3)
    input = [Variable(W), Variable(x)]
    ######################### test gmul ##############################
    # feature_maps = [num_features, num_features, num_features]
    # out = gmul(input)
    # print(out[0, :, num_features:])
    ######################### test gconv ##############################
    # feature_maps = [num_features, num_features, num_features]
    # gconv = Gconv(feature_maps, J)
    # _, out = gconv(input)
    # print(out.size())
    ######################### test gnn ##############################
    # x = torch.ones((bs, N, 1))
    # input = [Variable(W), Variable(x)]
    # gnn = GNN(num_features, num_layers, J)
    # out = gnn(input)
    # print(out.size())
    ######################### test siamese gnn ##############################
    x = torch.ones((bs, N, 1))
    input1 = [Variable(W), Variable(x)]
    input2 = [Variable(W.clone()), Variable(x.clone())]
    siamese_gnn = Siamese_GNN(num_features, num_layers, J)
    out = siamese_gnn(input1, input2)
    print(out.size())


