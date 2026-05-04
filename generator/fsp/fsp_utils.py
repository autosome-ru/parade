import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import seaborn as sns


CODES = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}
CODES_2 = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}

CELLTYPE_CODES_UTR3 = {
    "c1": 0, "c2": 1, "c4": 2, "c6": 3,
    "c17": 4, "c13": 5, "c10": 6,
}
CELLTYPE_CODES_UTR5 = {
    "c1": 0, "c2": 1, "c4": 2, "c6": 3, "c17": 4,
}


def id2n(n):
    return CODES[n]


def _get_celltype_codes(utr_type):
    if utr_type == "utr3":
        return CELLTYPE_CODES_UTR3
    elif utr_type == "utr5":
        return CELLTYPE_CODES_UTR5
    raise ValueError(f"Unknown utr_type: {utr_type!r}. Expected 'utr3' or 'utr5'.")


class Seq2Tensor(nn.Module):
    """One-hot encodes a nucleotide string into a (4, L) float tensor."""

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
    """Encodes a cell type label into a one-hot float vector."""

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


def gen_random_seqs(length, utr_type, cell_type, batch_size, device):
    """
    Generate a batch of random one-hot encoded sequences.

    Note: no fixed seed here - call np.random.seed() before if reproducibility
    is needed.
    """
    celltype_codes = _get_celltype_codes(utr_type)
    s2t = Seq2Tensor()
    seqs = ["".join(i) for i in np.random.choice(["A", "C", "G", "T"], size=(batch_size, length)).tolist()]
    out = torch.stack([s2t(seq) for seq in seqs], dim=0)
    return out.to(device)


def gen_random_seqs_no_logits(length, cell_type, batch_size):
    """Return a list of random nucleotide strings (no encoding)."""
    seqs = ["".join(i) for i in np.random.choice(["A", "C", "G", "T"], size=(batch_size, length)).tolist()]
    return seqs


def make_res(seqs_logits, cell_type, seq_len, utr_type, device):
    """
    Build model input tensors by concatenating:
    - sequence one-hot (4 channels)
    - reading frame channel (1 channel)
    - cell type condition channels (n_cell_types channels)

    Args:
        seqs_logits: Tensor of shape (batch, 4, L).
        cell_type: Cell type string key.
        seq_len: Sequence length (used for broadcasting condition channels).
        utr_type: "utr3" or "utr5".
        device: Torch device.

    Returns:
        Tensor of shape (batch, 4 + 1 + n_cell_types, L).
    """
    celltype_codes = _get_celltype_codes(utr_type)
    c2t = Condition2Tensor(num_conditions=len(celltype_codes), celltype_codes=celltype_codes)
    length = seqs_logits.shape[-1]
    out_seqs = torch.tensor([]).to(device)

    for seqs_logit in seqs_logits:
        seqs_logit = seqs_logit.to(device)
        frame = torch.tensor([1, 0, 0] * (length // 3 + 1), dtype=torch.float32).to(device)[:length]
        lines = c2t(cell_type).to(device)
        lines = lines[None, :, None].broadcast_to((1, len(celltype_codes), seq_len)).to(device)
        out = torch.concat([seqs_logit[None, ...], frame[None, None, ...], lines], dim=1)
        out_seqs = torch.concat([out_seqs, out])

    return out_seqs


def predict(seqs, model, cell_type, seq_len, utr_type, device):
    """Run the model on a batch of sequences for a given cell type."""
    with torch.no_grad():
        res = model(make_res(seqs, cell_type, seq_len, utr_type, device)).cpu()
    return res


def tensor_to_strings(x):
    """Convert a one-hot tensor (batch, 4, L) to a list of nucleotide strings."""
    idx2nt = ["A", "C", "G", "T"]
    x = x.argmax(dim=1)
    return ["".join(idx2nt[i.item()] for i in row) for row in x]


def predictor(seqs_batch, cell_type_target, model, seq_len, device, utr_type):
    """
    Predict activity (delta, index 1) for all cell types on a batch of sequences.

    Returns:
        pd.DataFrame with columns: seq, pred_scores, cell_type_pred, desired_cell_type.
    """
    celltype_codes = _get_celltype_codes(utr_type)
    seqs = []
    pred_scores = []
    cell_types_pred = []
    desired_cell_type_batch = []

    for cell in tqdm(celltype_codes):
        pred_score = predict(seqs_batch, model, cell, seq_len, utr_type, device)
        decoded_seq = seqs_batch.argmax(axis=1).cpu().numpy()
        encoded_seqs = ["".join(id2n(n) for n in seq) for seq in decoded_seq]
        seqs.extend(encoded_seqs)
        pred_scores.append(pred_score)
        cell_types_pred.append(np.full(pred_score.shape[0], cell))
        desired_cell_type_batch.append(np.full(pred_score.shape[0], cell_type_target))

    pred_df = pd.DataFrame({
        "seq": seqs,
        "pred_scores": torch.cat(pred_scores)[:, 1].numpy(),
        "cell_type_pred": np.concatenate(cell_types_pred),
        "desired_cell_type": np.concatenate(desired_cell_type_batch),
    })
    return pred_df


######################
##                  ##
##   Complexity     ##
##                  ##
######################


class _State:
    def __init__(self):
        self.len = 0
        self.link = -1
        self.next = dict()


def _build_suffix_automaton(s):
    sa = [_State()]
    last = 0
    size = 1
    total_substrings = 0

    for c in s:
        sa.append(_State())
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
                sa.append(_State())
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
    return _build_suffix_automaton(s)


def count_distinct_substrings_ratio(s):
    n = len(s)
    distinct = _build_suffix_automaton(s)
    max_possible = n * (n + 1) // 2
    return distinct / max_possible if max_possible != 0 else 0


######################
##                  ##
##   Tau metric     ##
##                  ##
######################


def calculate_tau_array(values):
    values_minus_1 = values - 1
    x_hat = values_minus_1 / np.max(values_minus_1)
    tau = np.sum(1 - x_hat) / (len(x_hat) - 1)
    return tau


def calculate_tau(df, excl=False, col_to_exclude=None):
    if col_to_exclude is None:
        col_to_exclude = ["max_diff_pair", "correlation", "subs", "max_diff", "seq", "max_diff_pair"]
    if excl:
        return df.drop(columns=col_to_exclude).apply(lambda row: calculate_tau_array(row.values), axis=1)
    return df.apply(lambda row: calculate_tau_array(row.values), axis=1)


######################
##                  ##
##    Plotting      ##
##                  ##
######################


def kdeplot_with_median(x, ax, color=None, fill=False, **kwargs):
    """KDE plot with a vertical dotted line at the median."""
    sns.kdeplot(x, fill=False, color=color, ax=ax, **kwargs)
    kdeline = ax.lines[-1]
    median = np.median(x)
    xs = kdeline.get_xdata()
    ys = kdeline.get_ydata()
    height = np.interp(median, xs, ys)
    ax.vlines(median, 0, height, color=color, ls=":")
    if fill:
        ax.fill_between(xs, 0, ys, facecolor=color, alpha=0.2)


######################
##                  ##
##  Inference DS    ##
##                  ##
######################


class UTRDataInference(Dataset):
    """
    Minimal dataset for running inference on a list of sequences.
    Builds model-ready tensors (seq + frame + condition channels).
    """

    def __init__(self, seqs, cell_type, utr_type, device):
        self.seqs = seqs
        self.cell_type = cell_type
        self.utr_type = utr_type
        self.device = device
        self.celltype_codes = _get_celltype_codes(utr_type)
        self.s2t = Seq2Tensor()
        self.c2t = Condition2Tensor(
            num_conditions=len(self.celltype_codes),
            celltype_codes=self.celltype_codes,
        )

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_logit = self.s2t(self.seqs[idx])
        seq_len = seq_logit.shape[-1]
        frame = torch.tensor([1, 0, 0] * (seq_len // 3 + 1), dtype=torch.float32)[:seq_len]
        lines = self.c2t(self.cell_type)
        lines = lines[:, None].broadcast_to((len(self.celltype_codes), seq_len))
        out = torch.concat([seq_logit, frame[None, ...], lines], dim=0)
        return out
