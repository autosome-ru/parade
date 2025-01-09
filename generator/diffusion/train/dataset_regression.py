#!/usr/bin/env python

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

from typing import Tuple


import numpy as np
import pandas as pd


CODES = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}

CELLTYPE_CODES = {"c1": 0,
                  "c2": 1,
                  "c4": 2,
                  "c6": 3,
                  "c17": 4,
                  "ex1": 5,
                  "ex2": 6,
                  "ex3": 7,
                  "ex4": 8}

UTR3_PREFIX = ("CAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAGGTGAACTTCAA"
               "GATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAGAACAC"
               "CCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGC"
               "CCTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGC"
               "CGCCGGGATCACTCTCGGCATGGACGAGCTGTACACTGCTAGCTAGATGACTAAACGCGT")
UTR3_SUFFIX = ("TTAATTAAACCGGACCGGTGGATCCAGACCACCTCCCCTGCGAGCTAAGCTGGACAGCCA"
               "ATGACGGGTAAGAGAGTGACATTTTTCACTAACCTAAGACAGGAGGGCCGTCAGAGCTAC"
               "TGCCTAATCCAAAGACGGGTAAAAGTGATAAAAATGTATCACTCCAACCTAAGACAGGCG"
               "CAGCTTCCGAGGGATTTGAGATCCAGACATGATAAGATACATTGATGAGTTTGGACAAAC"
               "CAAAACTAGAATGCAGTGAAAAAAATGCCTTATTTGTGAAATTTGTGATGCTATTGCCTT")
UTR5_PREFIX = ("GCAAGGAACCTTCCCGACTTAGGGGCGGAGCAGGAAGCGTCGCCGGGGGGCCCACAAGGG"
               "TAGCGGCGAAGATCCGGGTGACGCTGCGAACGGACGTGAAGAATGTGCGAGACCCAGGGT"
               "CGGCGCCGCTGCGTTTCCCGGAACCACGCCCAGAGCAGCCGCGTCCCTGCGCAAACCCAG"
               "GGCTGCCTTGGAAAAGGCGCAACCCCAACCCCGTGGGAATTCGATATCAAGCTTCTCGAG"
               "GGTAGGCGTGTACGGTGGGAGGCCTATATAAGCAGAGCTCGTTTAGTGAACCGTCAGATC")
UTR5_SUFFIX = ("GCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAG"
               "CTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCC"
               "ACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCCGTGCCCTGG"
               "CCCACCCTCGTGACCACCCTGACCTACGGCGTGCAGTGCTTCAGCCGCTACCCCGACCAC"
               "ATGAAGCAGCACGACTTCTTCAAGTCCGCCATGCCCGAAGGCTACGTCCAGGAGCGCACC")


class Seq2Tensor(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def n2id(n):
        return CODES[n.upper()]

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
    def __init__(self, num_conditions):
        super().__init__()
        self.num_conditions = num_conditions

    def forward(self, condition):
        if isinstance(condition, torch.FloatTensor):
            return condition
        code = CELLTYPE_CODES[condition]
        code = torch.tensor(code)
        code = F.one_hot(code, num_classes=self.num_conditions)

        return code.float()


def coin_toss(p=0.5):
    return torch.bernoulli(torch.tensor([p])).bool().item()


class UTRData(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        num_conditions: int = len(CELLTYPE_CODES),
        augment: bool = False,
        augment_test_time: bool = False,
        augment_kws: dict = None,
        features: tuple = ("sequence",),
        construct_type: str = "utr5",
    ):
        """
        :param: augment_kws (dict): a dictionary with any of the following keys:
            * shift_left (int): a maximal shift length to the left
            * shift_right (int): a maximal shift length to the right
            * extend_left (int): obligatory extension of the sequence to the left
            * extend_right (int): obligatory extension of the sequence to the right
            * revcomp (bool): whether to perform reverse-complement augmentation
        :param: features (tuple): a tuple of features in selected order. Possible values:
            * 'sequence': the channels for A, T, G, and C.
            * 'conditions': the channels for conditions.
            * 'positional': positional encoding feature.
            * 'revcomp': reverse-complement channel.
        """
        self.data = df

        self.num_conditions = num_conditions
        self.features = features
        self.num_channels = self.calculate_num_channels()

        # Augmentation options
        self.augment = augment or augment_test_time
        self.augment_test_time = augment_test_time
        self.augment_kws = augment_kws

        self.construct_type = construct_type.lower()
        if self.construct_type not in {"utr3", "utr5"}:
            raise ValueError('``construct_type`` must be either "utr3" or "utr5"')
        elif self.construct_type == "utr3":
            self.prefix = UTR3_PREFIX
            self.suffix = UTR3_SUFFIX
        elif self.construct_type == "utr5":
            self.prefix = UTR5_PREFIX
            self.suffix = UTR5_SUFFIX

        self.s2t = Seq2Tensor()
        self.c2t = Condition2Tensor(num_conditions=self.num_conditions)

        self.prepare_data(df)
        self.extend_seqs()
        self.encoded_seqs = torch.stack([self.s2t(seq) for seq in self.seqs.to_numpy()])

    def calculate_num_channels(self):
        n_ch = 0
        options = {"sequence": 4,
                   "intensity": 1,
                   "conditions": self.num_conditions,
                   "positional": 1,
                   "revcomp": 1}
        for k in self.features:
            n_ch += options[k]
        return n_ch

    def prepare_data(self, df: pd.DataFrame):
        self.seqs = df['seq']
        self.value = df['mass_center'].to_numpy(dtype=np.float32)
        self.celltype = torch.stack([self.c2t(c) for c in df['cell_type'].to_numpy()])
        self.replicate = torch.from_numpy(df['replicate'].to_numpy())

    def extend_seqs(self):
        shift_left = self.augment_kws["shift_left"]
        shift_right = self.augment_kws["shift_right"]
        extend_left = min(len(self.prefix), self.augment_kws["extend_left"])
        extend_right = min(len(self.suffix), self.augment_kws["extend_right"])

        self.seqlen = len(self.seqs.iloc[0]) + extend_left + extend_right

        extension_left = shift_left + extend_left
        extension_right = shift_right + extend_right
        if extension_left != 0:
            prefix = self.prefix[-extension_left:]
        else:
            prefix = ""
        suffix = self.suffix[:extension_right]

        def extseq(seq):
            return prefix + seq + suffix

        self.seqs = self.seqs.apply(extseq)
        self.flank_lengths = (shift_left, shift_right)

    def augment_seq(self, seq):
        if not self.augment:
            shift = 0
            toss = False
            return seq, shift, toss
        left, right = self.flank_lengths
        shift = torch.randint(low=-left, high=right + 1, size=tuple()).item()
        seq_shifted = seq[:, left + shift:left + self.seqlen + shift]
        if self.augment_kws["revcomp"]:
            toss = coin_toss()
            if toss:
                seq_shifted = self.revcomp_seq_tensor(seq_shifted)
        else:
            toss = False
        return seq_shifted, shift, toss

    def get_all_augments(self, seq):
        left, right = self.flank_lengths
        shifts = torch.arange(-left, right + 1)
        augms = torch.stack([seq[:, left + shift:left + self.seqlen + shift] for shift in shifts])
        tosses = torch.zeros(augms.shape[0], dtype=bool)
        if self.augment_kws["revcomp"]:
            shifts = torch.concat((shifts, shifts))
            augms = torch.concat((augms, self.revcomp_seq_tensor(augms, batch=True)))
            tosses = torch.concat((tosses, ~tosses))
        return augms, shifts, tosses

    @staticmethod
    def revcomp_seq_tensor(seq, batch=False):
        if batch:
            return torch.flip(seq, (1, 2))
        return torch.flip(seq, (0, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq = self.encoded_seqs[index]
        condition = self.celltype[index]
        value = self.value[index]

        if self.augment_test_time:
            seqs_augments_batch, shifts_batch, revcomp_batch = self.get_all_augments(seq)
            shift_correction = (shifts_batch * (-2 * revcomp_batch + 1) +
                                (1 - self.seqlen) * revcomp_batch)
            positional_batch = (shift_correction[:, None, None] + torch.arange(0, self.seqlen)) % 3 == 0
            revcomp_batch = revcomp_batch[:, None, None].broadcast_to((seqs_augments_batch.shape[0],
                                                                       1,
                                                                       self.seqlen))
            condition_batch = condition[None, :, None].broadcast_to((seqs_augments_batch.shape[0],
                                                                     self.num_conditions,
                                                                     self.seqlen))
            elements = {
                'sequence': seqs_augments_batch,
                'positional': positional_batch,
                'revcomp': revcomp_batch,
                'conditions': condition_batch,
            }
            to_concat = [elements[k] for k in self.features]
            seq_batch = torch.concat(to_concat, dim=1)
            return seq_batch, value
        else:
            seq, shift, revcomp = self.augment_seq(seq)
            if revcomp:
                positional_ch = (-shift - self.seqlen + 1 + torch.arange(0, self.seqlen)[None, :]) % 3 == 0
            else:
                positional_ch = (shift + torch.arange(0, self.seqlen)[None, :]) % 3 == 0
            revcomp_ch = torch.full((1, self.seqlen), fill_value=revcomp)
            condition_chs = condition[:, None].broadcast_to((condition.shape[0], seq.shape[-1]))
            elements = {
                'sequence': seq,
                'positional': positional_ch,
                'revcomp': revcomp_ch,
                'conditions': condition_chs,
            }
            to_concat = [elements[k] for k in self.features]
            compiled_seq = torch.concat(to_concat)
            return compiled_seq, value


class DataLoaderWrapper:
    def __init__(
        self,
        dataloader: DataLoader,
        batch_per_epoch: int,
    ):
        self.dataloader = dataloader
        self.batch_per_epoch = batch_per_epoch
        self.iterator = iter(self.dataloader)

    def __len__(self):
        return self.batch_per_epoch

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

    def __iter__(self):
        return self

    def reset(self):
        self.iterator = iter(self.dataloader)
        
        
        
MUT_POOLS = {0: (3, 1, 2, 0),
             3: (0, 1, 2, 3),
             1: (0, 3, 2, 1),
             2: (0, 3, 1, 2)}

def n2id(n):
    return CODES[n.upper()]

class PromotersData(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 limits: Tuple[int],
                 device = torch.device):
        self.device = device
        self.limits = limits
        self.data = df
        self.seqs = self.data['seq']
        self.score = self.data['mass_center']
        self.dataframe = df
        
    def __len__(self):
        return len(self.dataframe)
    
    
    def __getitem__(self, index):
        seq = self.seqs[index]
        seq_list = [n2id(i) for i in seq]
        code_target = torch.from_numpy(np.array(seq_list))
        code_target = F.one_hot(code_target, num_classes=5)
        nucl_sum = torch.sum(code_target,  dim=0)
        code_target[code_target[:, 4] == 1] = 0.25
        seqs_target_encode = (code_target[:, :4].float()).transpose(0, 1)        
        
        mut_num = torch.randint(low=self.limits[0], high=self.limits[1], size=(1,))

        mutation_sites = torch.randint(low=0, high=len(seq_list), size=(mut_num,))
        mutation_sites_bool_mask = torch.zeros(len(seq_list)+1)
        for site in mutation_sites:
            mutation_sites_bool_mask[site.to(torch.int64)] = 1 
        
        for site in mutation_sites:
            x = torch.randint(low=0, high=4, size=(1,))
            seq_list[site] = MUT_POOLS[seq_list[site]][x]
            
        mutated_seq = torch.from_numpy(np.array(seq_list))
        mutated_seq = F.one_hot(mutated_seq, num_classes=5)
        mutated_seq[mutated_seq[:, 4] == 1] = 0.25
        mutated_seq = (mutated_seq[:, :4].float()).transpose(0, 1)        
        seqs_mut_encode = torch.concat((
                                        mutated_seq,
                                        torch.full((1,mutated_seq.shape[1]), self.score[index]),
                                        torch.full((1,mutated_seq.shape[1]), mut_num[0]),
                                        torch.full((1,mutated_seq.shape[1]), nucl_sum[0]/len(seq_list)),
                                        torch.full((1,mutated_seq.shape[1]), nucl_sum[1]/len(seq_list)),
                                        torch.full((1,mutated_seq.shape[1]), nucl_sum[2]/len(seq_list)),
                                        torch.full((1,mutated_seq.shape[1]), nucl_sum[3]/len(seq_list)),
                                        
                                        ))

        return seqs_target_encode, seqs_mut_encode, mutation_sites_bool_mask


