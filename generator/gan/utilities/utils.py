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
from scipy.stats import pearsonr, spearmanr
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import seaborn as sns

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.distributions as dist
from torch.autograd import grad
import pytorch_lightning as pl
from IPython.display import clear_output

import utrdata_cl as utrdata
from pl_regressor import RNARegressor
from train_predictor import launch_model
from legnet_softclass import LegNetClassifier



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

CELLTYPE_CODES_UTR3 = {"c1": 0,
                       "c2": 1,
                       "c4": 2,
                       "c6": 3,
                       "c17": 4,
                       "c13": 5,
                       "c10": 6}

CELLTYPE_UTR3 = {"c1": 0,
                       "c2": 1,
                       "c4": 2,
                       "c6": 3,
                       "c17": 4,
                       "c13": 5}

CELLTYPE_UTR5 = {"c1": 0,
                       "c2": 1,
                       "c4": 2,
                       "c6": 3,
                       "c17": 4}


CELLTYPE_CODES_UTR5 = {"c1": 0,
                       "c2": 1,
                       "c4": 2,
                       "c6": 3,
                       "c17": 4}

def id2n(n):
    return CODES[n]



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


c2t = Condition2Tensor(num_conditions=len(CELLTYPE_CODES_UTR3), celltype_codes=CELLTYPE_CODES_UTR3)    


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


def make_res(seqs_logits, cell_type, seq_len, device):
    lengh = seqs_logits.shape[-1]
    out_seqs = torch.tensor([]).to(device)
    for seqs_logit in seqs_logits:
        seqs_logit = seqs_logit.to(device)
        frame = torch.tensor([1, 0, 0] * int((lengh / 3)), dtype=torch.float32).to(device)
        lines = c2t(cell_type).to(device)
        lines = lines[None, :, None].broadcast_to((1,
                                                  len(CELLTYPE_CODES_UTR3),
                                                  seq_len)).to(device)
        out = torch.concat([seqs_logit[None, ...], frame[None, None, ...], lines], dim=1)
        out_seqs = torch.concat([out_seqs, out])
    return out_seqs

def predict(seqs, model, cell_type, seq_len, device):
    with torch.no_grad():
        res = model(make_res(seqs, cell_type, seq_len, device)).cpu().detach()
    return res


def calculate_tau_array(values):
    values_minus_1 = values - 1
    x_hat = values_minus_1 / np.max(values_minus_1)
    tau = np.sum((1 - x_hat)) / (len(x_hat) - 1)
    return tau

def calculate_tau(df, excl = False, col_to_exclude = ['max_diff_pair', 'correlation', 'subs', 'max_diff', 'seq', 'max_diff_pair',]):
    if excl:
        tau_results = df.drop(columns=col_to_exclude).apply(lambda row: calculate_tau_array(row.values), axis=1)
    else:
        tau_results = df.apply(lambda row: calculate_tau_array(row.values), axis=1)
    return tau_results


def predictor(seqs_batch, cell_type_target, model, seq_len, device):
    seqs = []
    pred_scores = []
    cell_types_pred = []
    desired_cell_type_batch= []
    for cell in tqdm(CELLTYPE_UTR3):
        pred_score = predict(seqs_batch, model, cell, seq_len, device)
        decoded_seq = seqs_batch.argmax(axis=1).cpu().numpy()
        encoded_seqs = [''.join([id2n(n) for n in seq]) for seq in decoded_seq]
        seqs.extend(encoded_seqs)
        pred_scores.append(pred_score)
        cell_types_pred.append(np.full((pred_score.shape[0]), cell))
        desired_cell_type_batch.append(np.full((pred_score.shape[0]), cell_type_target))

    print('pred_scores', torch.cat(pred_scores)[:, 1].shape)
    print('cell_type_pred', np.concatenate(cell_types_pred).shape)
    print('desired_cell_type', np.concatenate(desired_cell_type_batch).shape)
    pred_df = pd.DataFrame().from_dict({"seq": seqs, 
                                        "pred_scores": torch.cat(pred_scores)[:, 1], 
                                        "cell_type_pred": np.concatenate(cell_types_pred), 
                                        "desired_cell_type": np.concatenate(desired_cell_type_batch)})
    return pred_df


def kdeplot_with_median(x, ax, color=None, fill=False, **kwargs):
    sns.kdeplot(x, fill=False, color=color, ax=ax, **kwargs)
    kdeline = ax.lines[-1]
    median = np.median(x)
    xs = kdeline.get_xdata()
    ys = kdeline.get_ydata()
    height = np.interp(median, xs, ys)
    ax.vlines(median, 0, height, color=color, ls=':')
    if fill:
        ax.fill_between(xs, 0, ys, facecolor=color, alpha=0.2)


class UTRData(Dataset):
    def __init__(self, seqs: str, cell_type: str, UTR: str, device):
        self.UTR = UTR
        self.seqs = seqs
        self.cell_type = cell_type
        self.device = device
        
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



def calculate_center_of_mass(df):
    # Group by seq, cell_type, replicate and sum
    grouped_df = df.groupby(['seq', 'cell_type', 'replicate']).sum().reset_index()
    
    # Calculate center of mass using bins 1-4
    bins = np.arange(1, 5)
    cpm = grouped_df[["1", "2", "3", "4"]]
    grouped_df['center_of_mass'] = (cpm * bins).sum(axis=1) / cpm.sum(axis=1)
    # Average center of mass across replicates
    final_df = grouped_df.groupby(['seq', 'cell_type'])['center_of_mass'].mean().reset_index()
    
    # Pivot to get cell types as columns
    result_df = final_df.pivot(
        index='seq',
        columns='cell_type',
        values='center_of_mass'
    )
    
    return result_df

def calculate_correlations_of_patterns(df1_, df2_, method='spearman'):
    # Create deep copies of both dataframes to avoid modifying originals
    df1 = df1_.copy(deep=True)#.head(n = 10)
    df2 = df2_.copy(deep=True)
    # Get numerical columns present in both dataframes
    df1_num_cols = df1.select_dtypes(include=['float64', 'int64']).columns
    df2_num_cols = df2.select_dtypes(include=['float64', 'int64']).columns
    
    # Find overlapping columns
    common_cols = list(set(df1_num_cols).intersection(set(df2_num_cols)))
    print(f"Common numerical columns: {common_cols}")
    
    if not common_cols:
        raise ValueError("No common numerical columns found between the dataframes.")
    
    # Convert index and seq to uppercase before merging
    df1.index = df1.index.str.upper()
    df2['seq'] = df2['seq'].str.upper()
    # Merge dataframes using left index and right seq column
    merged_df = pd.merge(df1, df2, left_index=True, right_on='seq', how='inner', suffixes=('_df1', '_df2'))
    print(f"Number of sequences after merging: {len(merged_df)}")
    
    def calc_corr(row, common_cols, method):
        values1 = row[[col + '_df1' for col in common_cols]]#.dropna()
        values2 = row[[col + '_df2' for col in common_cols]]#.dropna()
        values1 = values1.tolist()
        values2 = values2.tolist()
        # Convert lists to numpy arrays for correlation calculation
        values1 = np.array(values1)
        values2 = np.array(values2)
        
        # Check if we have enough values
        if len(values1) < 2 or len(values2) < 2:
            print(f"Insufficient values for sequence: {row['seq']}")
            return np.nan
            
        # Calculate correlation using numpy
        if method == 'spearman':
            corr = np.corrcoef(values1, values2)[0,1]
        else: # pearson
            corr = np.corrcoef(values1, values2)[0,1]
            
        return corr
    
    # Calculate correlations
    correlations = merged_df.apply(
        lambda row: {'seq': row['seq'], 
                     'correlation': calc_corr(row, common_cols, method)},
        axis=1
    ).tolist()
    
    result_df = pd.DataFrame(correlations).dropna()
    print(result_df.shape)
    
    
    return result_df

def plot_expression_radar(df,
                          title='',
                          UTR = '5',
                          cols_to_draw=False,
                          cols_to_drop=['seq','tau', 'Cluster', 'Unified_Cluster',
                                  'subs', 'max_diff_x', 'max_diff_pair_pred', 'source', 'max_diff_y', 'max_diff_pair_exp']):
    """
    Create radar plots for each row in the dataframe showing expression levels across cell types.
    
    Args:
        df: DataFrame containing expression data, with columns for cell types and metadata
            (tau, Cluster, Unified_Cluster)
    
    Returns:
        matplotlib figure object
    """
    # Create radial plots for each representative sequence
    n_rows = len(df)
    fig, axes = plt.subplots(1, n_rows, figsize=(4*n_rows, 4), subplot_kw=dict(projection='polar'))
    fig.suptitle(f'{title}')
    # Handle case of single row dataframe
    if n_rows == 1:
        axes = [axes]
    
    # Get columns to plot (cell types)
    if cols_to_draw:
        cols_to_plot = cols_to_draw
    else:
        cols_to_plot = [col for col in df.columns 
                        if col not in cols_to_drop]
    n_cols = len(cols_to_plot)
    
    # Calculate angles for the plot
    angles = np.linspace(0, 2*np.pi, len(cols_to_plot), endpoint=False)
    # Close the plot by appending first value
    angles = np.concatenate((angles, [angles[0]]))
    
    # Get global min and max for consistent scale
    min_val = 1
    max_val = 4
    
    for idx, (_, row) in enumerate(df.iterrows()):
        # Get values and close the plot by appending first value
        values = [row[col] for col in cols_to_plot]
        values = np.concatenate((values, [values[0]]))
        
        # Plot
        axes[idx].plot(angles, values)
        axes[idx].fill(angles, values, alpha=0.25)
        if title:
            axes[idx].set_title(f'{title}')
        # Set the labels and limits
        axes[idx].set_xticks(angles[:-1])
        axes[idx].set_xticklabels(cols_to_plot)
        axes[idx].set_ylim(min_val, max_val)
        if UTR == '5':
            print(df['seq'][idx])
            axes[idx].set_title(f"r={round(df['correlation'][idx], 2)}, tau={round(df['tau'][idx], 2)}, min_dist={df['min_dist'][idx]}, {df['source'][idx]} \n {df['seq'][idx]}", fontsize=6)
        else:
            print(idx, df['seq'][idx])
            axes[idx].set_title(f"r={round(df['correlation'][idx], 2)}, tau={round(df['tau'][idx], 2)}, min_dist={df['min_dist'][idx]}, {df['source'][idx]} ", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_expression_barplot(df, cols_to_draw):
    """
    Create bar plots for each row in the dataframe showing expression levels across cell types.
    
    Args:
        df: DataFrame containing expression data, with columns for cell types and metadata
            (tau, Cluster, Unified_Cluster)
    
    Returns:
        matplotlib figure object
    """
    # Create bar plots for each representative sequence
    n_rows = len(df)
    fig, axes = plt.subplots(1, n_rows, figsize=(4*n_rows, 4))
    
    # Handle case of single row dataframe
    if n_rows == 1:
        axes = [axes]
    
    # Get columns to plot (cell types)
    cols_to_plot = cols_to_draw
    n_cols = len(cols_to_plot)
    
    # Get global min and max for consistent y-axis
    y_min = 1.2
    y_max = 3.8
    
    x = np.arange(len(cols_to_plot))
    
    for idx, (_, row) in enumerate(df.iterrows()):
        # Get values
        values = [row[col] for col in cols_to_plot]
        
        # Plot
        axes[idx].bar(x, values)
        
        # Set the labels
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(cols_to_plot, rotation=45)
        axes[idx].set_ylim(y_min, y_max)
        if 'Unified_Cluster' in row:
            axes[idx].set_title(f'Sequence {idx+1}\nCluster: {row["Unified_Cluster"]}')
        else:
            axes[idx].set_title(f'Sequence {idx+1}')
    
    plt.tight_layout()
    plt.show()
