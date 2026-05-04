import shutil
import sys
import os
import csv
import time
import math
import importlib
import tempfile

import itertools
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import pearsonr
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributions as dist
from torch.autograd import grad
import pytorch_lightning as pl
from IPython.display import clear_output

import utilities.utrdata_cl as utrdata
from utilities.pl_regressor import RNARegressor
from utilities.train_predictor import launch_model
from utilities.legnet_softclass import LegNetClassifier

from utilities.UTR5_utilities import Seq2Tensor, Condition2Tensor, make_res, CELLTYPE_UTR5


def gen_random_seqs(lengh, cell_type, batch_size, device):
    s2t = Seq2Tensor()
    np.random.seed(7)
    
    seq_1 = [''.join(i) for i in np.random.choice(['A', 'C', 'G', 'T'], size=(batch_size, lengh)).tolist()]
    out = torch.stack([s2t(seq) for seq in seq_1], dim=0)
    return out.to(device)

def gen_random_seqs_no_logits(lengh, cell_type, batch_size):
    np.random.seed(7)
    
    seq_1 = [''.join(i) for i in np.random.choice(['A', 'C', 'G', 'T'], size=(batch_size, lengh)).tolist()]
    return seq_1


######## complexity score count utils ########
class State:
    def __init__(self):
        self.len = 0
        self.link = -1
        self.next = dict()

def build_suffix_automaton(s):
    sa = [State()]
    last = 0
    size = 1
    total_substrings = 0

    for c in s:
        current = last
        sa.append(State())
        new_state = size
        sa[new_state].len = sa[last].len + 1
        size += 1

        p = last
        while p != -1 and c not in sa[p].next:
            sa[p].next[c] = new_state
            p = sa[p].link

        if p == -1:
            sa[new_state].link = 0
        else:
            q = sa[p].next[c]
            if sa[p].len + 1 == sa[q].len:
                sa[new_state].link = q
            else:
                clone = size
                sa.append(State())
                sa[clone].len = sa[p].len + 1
                sa[clone].next = sa[q].next.copy()
                sa[clone].link = sa[q].link
                size += 1
                while p != -1 and sa[p].next[c] == q:
                    sa[p].next[c] = clone
                    p = sa[p].link
                sa[q].link = clone
                sa[new_state].link = clone
        last = new_state
        total_substrings += sa[new_state].len - sa[sa[new_state].link].len

    return total_substrings

def count_distinct_substrings(s):
    return build_suffix_automaton(s)


def count_distinct_substrings_ratio(s):
    n = len(s)
    distinct_substrings = build_suffix_automaton(s)
    max_possible_substrings = n * (n + 1) // 2
    ratio = distinct_substrings / max_possible_substrings if max_possible_substrings != 0 else 0
    return ratio



def tensor_to_strings(x):
    idx2nt = ['A', 'C', 'G', 'T']
    x = x.argmax(dim=1) 
    seqs = []
    for row in x:
        seq = ''.join([idx2nt[i.item()] for i in row])
        seqs.append(seq)
    return seqs

class BaseEnergy(torch.nn.Module):
    """
    BaseEnergy class for defining energy functions in sequence optimization.
    This class serves as a base class for defining energy functions to be used in sequence optimization.
    Subclasses should implement the `energy_calc` method to compute the energy of input sequences.

    Methods:
        __init__(): Initialize the BaseEnergy class.
        forward(x_in): Compute the energy of input sequences and apply penalties if applicable.
        energy_calc(x): Calculate the energy of input sequences.

    Note:
        - Subclasses must implement the `energy_calc` method to compute energy.

    """
    
    def __init__(self):
        """
        Initialize the BaseEnergy class.

        Note:
            This constructor initializes the model attribute to None.
        """
        super().__init__()

        self.model = None
        
    def forward(self, x_in):
        """
        Compute the energy of input sequences and apply penalties if applicable.

        Args:
            x_in (torch.Tensor): Input sequences.

        Returns:
            torch.Tensor: Computed energy values.

        """
        hook = self.energy_calc(x_in)
        # pen = self.penalty(x_in)
        # try:
        #     pen = self.penalty(x_in)
        #     hook = hook + pen
        #     print(pen)
        # except AttributeError:
        #     try:
        #         _ = self.you_have_been_warned
        #     except AttributeError:
        #         print("Penalty not implemented", file=sys.stderr)
        #         self.you_have_been_warned = True
        #     pass
        return hook 
      
    def energy_calc(self, x):
        """
        Calculate the energy of input sequences.

        Args:
            x (torch.Tensor): Input sequences.

        Raises:
            NotImplementedError: Raised when the method is not implemented.

        Returns:
            torch.Tensor: Computed energy values.

        """
        raise NotImplementedError("Energy caclulation not implemented.")
        x_in = x.to(self.model.device)
        
        hook = self.model(x_in)
        # do math

        return hook
      

class MinGapEnergy(BaseEnergy):
    """
    MinGapEnergy class for defining energy functions based on MinGap values. This is a replica
    of OverMaxEnergy that better reflects the final terminology in the manuscript.

    This class inherits from BaseEnergy and defines an energy function that calculates
    the gap between the target (bias) cell activity and the maximum off-target (non-bias) cell
    activity of a model's output for input sequences, with an optional value-bending factor.

    Args:
        model (torch.nn.Module): The neural network model used for energy calculation.
        
        
         
           (int, optional): Index of the target feature prediction. Default is 0.
        target_alpha (float, optional): Scaling factor for the bias term. Default is 1.0.
        bending_factor (float, optional): Bending factor applied to the model's output. Default is 0.0.
        a_min (float, optional): Minimum value allowed after bending. Default is negative infinity.
        a_max (float, optional): Maximum value allowed after bending. Default is positive infinity.

    Methods:
        add_energy_specific_args(parent_parser): Add energy-specific arguments to an argparse ArgumentParser.
        process_args(grouped_args): Process grouped arguments and return energy-related arguments.
        bend(x): Apply bending factor to the input tensor.
        energy_calc(x): Calculate the energy of input sequences based on the maximum model outputs.

    Note:
        - The `model` provided must be a neural network model compatible with PyTorch.

    """
    
    @staticmethod
    def add_energy_specific_args(parent_parser):
        """
        Add energy-specific arguments to an argparse ArgumentParser.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added energy-specific arguments.

        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Energy Module args')
        group.add_argument('--target_feature', type=int, default=0)
        group.add_argument('--target_alpha', type=float, default=1.)
        group.add_argument('--bending_factor', type=float, default=0.)
        group.add_argument('--a_min', type=float, default=-math.inf)
        group.add_argument('--a_max', type=float, default=math.inf)
        return parser

    @staticmethod
    def process_args(grouped_args):
        """
        Process grouped arguments and return energy-related arguments.

        Args:
            grouped_args (dict): Grouped arguments.

        Returns:
            dict: Processed energy-related arguments.

        """
        energy_args = grouped_args['Energy Module args']
        
        model.to(device)
        model.eval()
        energy_args.model = model

        del energy_args.model_artifact
        
        return energy_args

    def __init__(self, model, device, seq_len, target_feature=0, target_alpha=1., bending_factor=0., a_min=-math.inf, a_max=math.inf, loss_type='malinois'):
        """
        Initialize the MinGapEnergy class.

        Args:
            model (torch.nn.Module): The neural network model used for energy calculation.
            target_feature (int, optional): Index of the target feature. Default is 0.
            target_alpha (float, optional): Scaling factor for the target feature. Default is 1.0.
            bending_factor (float, optional): Bending factor applied to the model's output. Default is 0.0.
            a_min (float, optional): Minimum value allowed after bending. Default is negative infinity.
            a_max (float, optional): Maximum value allowed after bending. Default is positive infinity.

        """
        super().__init__()
        
        self.model = model
        self.model.eval()
        self.loss_type = loss_type

        self.target_feature = target_feature
        self.target_alpha= target_alpha
        self.bending_factor = bending_factor
        self.a_min = a_min
        self.a_max = a_max
        self.device = device
        self.seq_len = seq_len

    def bend(self, x):
        """
        Apply bending to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with applied bending.

        """
        return x - self.bending_factor * (torch.exp(-x) - 1)
        
    def energy_calc(self, x):
        """
        Calculate the energy of input sequences based on the maximum model outputs.

        Args:
            x (torch.Tensor): Input sequences.

        Returns:
            torch.Tensor: Computed energy values.

        """
        
        x = x.to(self.device)
        batch_hook = torch.tensor([]).to(self.device)
        for cell_line in CELLTYPE_UTR5:
            hook = make_res(x, cell_line, device=self.device).to(self.device)
            hook = self.bend(self.model(hook).clamp(self.a_min,self.a_max)).to(self.device)
            hook = hook[:, 1].unsqueeze(dim=0)
            batch_hook = torch.concat([batch_hook, hook]).to(self.device)
        
        hook = batch_hook.T.to(self.device)
        
        if self.loss_type == 'malinois':
            energy = hook[...,[x for x in range(hook.shape[-1]) if x != self.target_feature]].mean(-1) \
                 - hook[...,self.target_feature].mul(self.target_alpha)
            
        elif self.loss_type == 'square': 
            energy = -(hook[...,[x for x in range(hook.shape[-1]) if x != self.target_feature]].mean(-1) \
                 - hook[...,self.target_feature] )** 2
            
        elif self.loss_type == 'square_adj':
            energy = -((hook[...,[x for x in range(hook.shape[-1]) if x != self.target_feature]]** 2).sum(-1) \
                 - (hook[...,self.target_feature] - hook[...,[x for x in range(hook.shape[-1]) if x != self.target_feature]].amax(-1))** 2)           
            
        elif self.loss_type == 'exp': 
            energy = (torch.exp(hook[...,[x for x in range(hook.shape[-1]) if x != self.target_feature]])).mean(dim=1) \
                 - (torch.exp(hook[...,self.target_feature]))            
            
        return energy

    # def penalty(self, x):
    #     x = x.to(device)
    #     strings = tensor_to_strings(x)  
    #     penalties = []

    #     for s in strings:
    #         ratio = count_distinct_substrings_ratio(s)
    #         penalties.append(ratio) 

    #     return torch.tensor(penalties, dtype=torch.float32, device=device)
