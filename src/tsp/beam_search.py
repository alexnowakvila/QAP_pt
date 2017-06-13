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
        self.mask = (torch.Tensor(batch_size, beam_size,
                     N).type(dtype).fill_(1))
        # mask the starting node of the beam search
        self.mask[:, :, 0] = 0
        self.done = False
        # The score for each translation on the beam.
        self.scores = torch.Tensor(batch_size, beam_size).type(dtype).zero_()
        self.All_scores = []
        # The backpointers at each time-step.
        self.prev_Ks = []
        # The outputs at each time-step.
        self.next_nodes = ([torch.Tensor(batch_size, beam_size)
                           .type(dtype_l).fill_(-1)])
        self.next_nodes[0][:, 0] = 0

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prev_Ks[-1]

    def advance(self, trans_probs, it):
        # prev_probs: probabilites of advancing from the next step
        """Advance the beam."""
        # trans_probs has size (bs, K, N)
        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = (trans_probs + self.scores.unsqueeze(2)
                       .expand_as(prev_probs))
        else:
            beam_lk = trans_probs[:, 0]
        beam_lk = beam_lk * self.mask
        beam_lk = beam_lk.view(self.batch_size, -1)
        # beam_lk has size (bs, K*N)
        bestScores, bestScoresId = beam_lk.topk(self.beam_size, 1, True, True)
        # bestScores and bestScoresId have size (bs, K)
        self.scores = bestScores
        prev_k = bestScoresId // self.N
        self.prev_Ks.append(prev_k)
        new_nodes = bestScoresId - prev_k * self.N
        self.next_nodes.append(new_nodes)
        # reindex mask
        perm_mask = prev_k.unsqueeze(2).expand_as(mask) # (bs, K, N)
        self.mask = self.mask.gather(1, perm_mask)
        # mask new added nodes
        self.update_mask(new_nodes)
        # End condition is when top-of-beam is EOS.
        if it == self.N-1:
            self.done = True
        return self.done

    def update_mask(self, new_nodes):
        # sets new_nodes to zero in mask
        arr = (torch.arange(0, self.N).unsqueeze(1).unsqueeze(0)
               .expand_as(self.mask))
        new_nodes = new_nodes.unsqueeze(2).expand_as(self.mask)
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
        hyp = []
        for j in range(len(self.prev_Ks) - 1, -1, -1):
            hyp.append(self.next_nodes[j + 1][k])
            k = self.prev_Ks[j][k]
        return hyp[::-1]