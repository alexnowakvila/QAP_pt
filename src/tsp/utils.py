#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
# import dependencies
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
    N = W.size()[-1]
    batch_size = W.size()[0]
    costs = []
    pred = torch.topk(pred, 2, dim=2)[1]
    for b in range(batch_size):
        cost = 0.0
        predb = pred[b]
        Wb = W[b]
        start = 0
        end = next_vertex(start, -1, predb)
        # print(start, end)
        for i in range(N-1):
            cost += Wb[start, end]
            prev = start
            start = end
            end = next_vertex(start, prev, predb)
            # print(start, end)
        cost += Wb[start, end]
        costs.append(cost.data.cpu().numpy())
    return costs

class BeamSearch(object):
    """Ordered beam of candidate outputs."""
    def __init__(self, beam_size, batch_size, pred, N):
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.N = N
        self.chosen = (torch.Tensor(batch_size, beam_size,
                       N).type(dtype).fill_(1))
        # mask the starting node of the beam search
        self.chosen[:, :, 0] = 0
        self.done = False
        # The score for each translation on the beam.
        self.scores = torch.Tensor(batch_size, beam_size).type(dtype).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = ([torch.Tensor(batch_size, beam_size)
                       .type(dtype_l).fill_(-1)])
        self.nextYs[0][0] = 0

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    def advance(self, prev_probs):
        # prev_probs: probabilites of advancing from the next step
        """Advance the beam."""
        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = (prev_probs + self.scores.unsqueeze(2)
                       .expand_as(prev_probs))
        else:
            beam_lk = prev_probs[:, 0]
        beam_lk = beam_lk * self.chosen
        bestScores, bestScoresId = beam_lk.topk(self.beam_size, 2, True, True)
        self.scores = bestScores
        # update mask
        mask = mask.gather(2, bestScoresId)
        prev_k = bestScoresId
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)
        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True
        return self.done

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1]



###############################################################################

"""Beam search implementation in PyTorch."""
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

# Code borrowed from PyTorch OpenNMT example
# https://github.com/pytorch/examples/blob/master/OpenNMT/onmt/Beam.py

import torch


class Beam(object):
    """Ordered beam of candidate outputs."""

    def __init__(self, size, vocab, cuda=False):
        """Initialize params."""
        self.size = size
        self.done = False
        self.pad = vocab['<pad>']
        self.bos = vocab['<s>']
        self.eos = vocab['</s>']
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

        # The attentions (matrix) for each time.
        self.attn = []

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.

    def advance(self, workd_lk):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True

        return self.done

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def get_hyp(self, k):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]

        return hyp[::-1]



