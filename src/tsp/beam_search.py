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

###############################################################################
#                               Beam Search                                   #
###############################################################################

class BeamSearch(object):
    """Ordered beam of candidate outputs."""
    def __init__(self, beam_size, batch_size, N):
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.N = N
        self.mask = torch.ones(batch_size, beam_size, N).type(dtype)
        # mask the starting node of the beam search
        self.mask[:, :, 0] = 0
        # The score for each translation on the beam.
        self.scores = torch.zeros(batch_size, beam_size).type(dtype)
        self.All_scores = []
        # The backpointers at each time-step.
        self.prev_Ks = []
        # The outputs at each time-step.
        self.next_nodes = [torch.zeros(batch_size, beam_size).type(dtype_l)]

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        current_state =  (self.next_nodes[-1].unsqueeze(2)
                          .expand(self.batch_size, self.beam_size, self.N))
        return current_state

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prev_Ks[-1]

    def advance(self, trans_probs, it):
        # prev_probs: probabilites of advancing from the next step
        """Advance the beam."""
        # trans_probs has size (bs, K, N)
        # Sum the previous scores.
        if len(self.prev_Ks) > 0:
            beam_lk = (trans_probs + self.scores.unsqueeze(2)
                       .expand_as(trans_probs))
        else:
            beam_lk = trans_probs
            # only use the first element of the beam (mask to zero the others)
            beam_lk[:, 1:] = torch.zeros(beam_lk[:, 1:].size()).type(dtype)
        beam_lk = beam_lk * self.mask
        beam_lk = beam_lk.view(self.batch_size, -1)
        # beam_lk has size (bs, K*N)
        bestScores, bestScoresId = beam_lk.topk(self.beam_size, 1, True, True)
        # bestScores and bestScoresId have size (bs, K)
        self.scores = bestScores
        prev_k = bestScoresId / self.N
        self.prev_Ks.append(prev_k)
        new_nodes = bestScoresId - prev_k * self.N
        self.next_nodes.append(new_nodes)
        # reindex mask
        perm_mask = prev_k.unsqueeze(2).expand_as(self.mask) # (bs, K, N)
        self.mask = self.mask.gather(1, perm_mask)
        # mask new added nodes
        self.update_mask(new_nodes)

    def update_mask(self, new_nodes):
        # sets new_nodes to zero in mask
        arr = (torch.arange(0, self.N).unsqueeze(0).unsqueeze(1)
               .expand_as(self.mask).type(dtype_l))
        new_nodes = new_nodes.unsqueeze(2).expand_as(self.mask)
        # print(arr, new_nodes)
        update_mask = 1 - torch.eq(arr, new_nodes).type(dtype)
        self.mask = self.mask*update_mask

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    def get_hyp(self, k):
        """ Walk back to construct the full hypothesis.
        k: the position in the beam to construct."""
        assert self.N == len(self.prev_Ks) + 1
        hyp = -1*torch.ones(self.batch_size, self.N).type(dtype_l)
        # first node always zero
        hyp[:, 0] = torch.zeros(self.batch_size, 1).type(dtype_l)
        for j in range(len(self.prev_Ks) - 1, -1, -1):
            hyp[:, j+1] = self.next_nodes[j + 1].gather(1, k)
            k = self.prev_Ks[j].gather(1, k)
        return hyp