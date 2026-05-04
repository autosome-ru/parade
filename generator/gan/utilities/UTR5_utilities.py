import sys
import os
import importlib

import itertools

import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import pearsonr
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import seaborn as sns

from tqdm.auto import tqdm as tqdm_auto
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributions as dist
from torch.autograd import grad

CODES = {
    0:"A",
    1:"C",
    2:"G",
    3:"T",
    4:"N",
}


CODES_2 = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}

CELLTYPE_UTR5 = {"c1": 0,
                       "c2": 1,
                       "c4": 2,
                       "c6": 3,
                       "c17": 4,}


CELLTYPE_CODES_UTR5 = {"c1": 0,
                       "c2": 1,
                       "c4": 2,
                       "c6": 3,
                       "c17": 4}

class Seq2Tensor(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def n2id(n):
        return CODES_2[n.upper()]

    def forward(self, seq):
        if isinstance(seq, torch.FloatTensor):
            return seq
        seq = [self.n2id(x) for x in seq.upper()]
        code = torch.tensor(seq)
        code = F.one_hot(code, num_classes=5)

        code[code[:, 4] == 1] = 0.25
        code = code[:, :4].float()
        return code.transpose(0, 1)
    
class Condition2Tensor(nn.Module):
    def __init__(self, num_conditions, celltype_codes):
        super().__init__()
        self.num_conditions = num_conditions
        self.celltype_codes = celltype_codes

    def forward(self, condition):
        if isinstance(condition, torch.FloatTensor):
            return condition
        code = self.celltype_codes[condition]
        code = torch.tensor(code)
        code = F.one_hot(code, num_classes=self.num_conditions)

        return code.float()
    

# def make_res(seqs_logits, cell_type, seq_len, device):
#     c2t = Condition2Tensor(
#         num_conditions=len(CELLTYPE_CODES_UTR5), 
#         celltype_codes=CELLTYPE_CODES_UTR5
#     )
#     lengh = seqs_logits.shape[-1]
#     out_seqs = torch.tensor([]).to(device)
#     for seqs_logit in seqs_logits:
#         seqs_logit = seqs_logit.to(device)
#         frame = torch.tensor(([1, 0, 0] * (int(lengh / 3) + 1))[:-1], dtype=torch.float32).to(device)
#         lines = c2t(cell_type).to(device)
#         lines = lines[None, :, None].broadcast_to((1,
#                                                   len(CELLTYPE_CODES_UTR5),
#                                                   seq_len)).to(device)
#         out = torch.concat([seqs_logit[None, ...], frame[None, None, ...], lines], dim=1)
#         out_seqs = torch.concat([out_seqs, out])
#     return out_seqs

def make_res(seqs_logits, cell_type, UTR='5UTR', device='cuda:0'):
        chan = len(CELLTYPE_CODES_UTR5)
        c2t = Condition2Tensor(num_conditions=len(CELLTYPE_CODES_UTR5), celltype_codes=CELLTYPE_CODES_UTR5)
        seq_len = seqs_logits.shape[-1]
        out_seqs = torch.tensor([]).to(device)
        for seqs_logit in seqs_logits:
            seqs_logit = seqs_logit.to(device)
            frame = torch.tensor([1, 0, 0]).repeat(seq_len // 3 + 1)[:seq_len].to(device)
            lines = c2t(cell_type).to(device)
            lines = lines[None, :, None].broadcast_to((1,
                                                      chan,
                                                      seq_len)).to(device)
            print(seqs_logit[None, ...].shape, frame[None, None, ...].shape, lines.shape)
            out = torch.concat([seqs_logit[None, ...], frame[None, None, ...], lines], dim=1)
            out_seqs = torch.concat([out_seqs, out])
        return out_seqs

def predict(seqs, model, cell_type, device):
    res = model(make_res(seqs, cell_type, device)).cpu().detach()[:,1]
    return res.cpu().numpy()


def seqprep(seq):
    s2t = Seq2Tensor()
    seq1 = torch.unsqueeze(s2t(seq), dim=0)
    return seq1

class UTRData(Dataset):
    def __init__(self, seqs, cell_type, UTR, device):
        self.UTR = UTR
        self.seqs = seqs
        self.device = device
        self.cell_type = cell_type
        
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self,idx):
        s2t = Seq2Tensor()     
        seqs_logits = torch.unsqueeze(s2t(self.seqs[idx]), dim=0)
        if self.UTR == '3UTR':
            chan = len(CELLTYPE_CODES_UTR3)
            c2t = Condition2Tensor(num_conditions=len(CELLTYPE_CODES_UTR3), celltype_codes=CELLTYPE_CODES_UTR3)
            seq_len = seqs_logits.shape[-1]
            out_seqs = torch.tensor([]).to(self.device)
            for seqs_logit in seqs_logits:
                seqs_logit = seqs_logit.to(self.device)
                frame = torch.tensor([1, 0, 0]).repeat(seq_len // 3 + 1)[:seq_len].to(self.device)
                lines = c2t(self.cell_type).to(self.device)
                lines = lines[None, :, None].broadcast_to((1,
                                                          chan,
                                                          seq_len)).to(self.device)
                out = torch.concat([seqs_logit[None, ...], frame[None, None, ...], lines], dim=1)
                out_seqs = torch.concat([out_seqs, out])
            return out_seqs.squeeze()

        elif self.UTR == '5UTR':
            chan = len(CELLTYPE_CODES_UTR5)
            c2t = Condition2Tensor(num_conditions=len(CELLTYPE_CODES_UTR5), celltype_codes=CELLTYPE_CODES_UTR5)
            seq_len = seqs_logits.shape[-1]
            out_seqs = torch.tensor([]).to(self.device)
            for seqs_logit in seqs_logits:
                seqs_logit = seqs_logit.to(self.device)
                frame = torch.tensor([1, 0, 0]).repeat(seq_len // 3 + 1)[:seq_len].to(self.device)
                lines = c2t(self.cell_type).to(self.device)

                lines = lines[None, :, None].broadcast_to((1,
                                                          chan,
                                                          seq_len)).to(self.device)

                out = torch.concat([seqs_logit[None, ...], frame[None, None, ...], lines], dim=1)
                out_seqs = torch.concat([out_seqs, out])
            return out_seqs.squeeze()
