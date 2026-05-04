import numpy as np
import pandas as pd
import torch

from pathlib import Path
from tqdm.auto import tqdm


idx2cell_type = ['c1', 'c2', 'c4', 'c6', 'c17', 'c13', 'c10']

cell_type_mapper = {
    'c1': 'MDA-MB-231',
    'c2': 'HepG2',
    'c4': 'Jurkat',
    'c6': 'SW480',
    'c13': 'PA-1',
    'c17': 'NALM6'
}

NUCL_ORDER = ('A', 'C', 'G', 'T')

class OneHotSeqEncoder:
    def __init__(self):
        eye = torch.eye(4)
        self.letter2tensor = dict(zip(NUCL_ORDER, eye))
        self.letter2tensor['N'] = torch.ones(4) * 0.25
        self.idx2letter = dict(zip(range(4), NUCL_ORDER))

    def __call__(self, sequence: str) -> torch.Tensor:
        nucleotides = torch.cat(
            [self.letter2tensor[nt][None] for nt in sequence], dim=0)
        return nucleotides.T  # not (4, -1), it will ruin the order and everything

    def decode(self, sequences: torch.Tensor) -> list[str]:
        """

        :param sequences: list of sequences encoded by token numbers
         according to `self.idx2letter`
        :return:
        """
        result = []
        for seq in sequences:
            result.append(''.join([self.idx2letter[idx.item()] for idx in seq]))
        return result


class UTRDataset:
    def __init__(
        self,
        sequences: list[str],
        ohe_sequences: list[torch.Tensor],
        cell_types: list[torch.Tensor],
        expressions: list[torch.Tensor],
        deltas: list[torch.Tensor]
    ) -> None:

        self.sequences = sequences
        self.ohe_sequences = ohe_sequences
        self.cell_types = cell_types
        self.expressions = expressions
        self.deltas = deltas

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            'ohe_seq': self.ohe_sequences[idx],
            'cell_type': self.cell_types[idx],
            'expression': self.expressions[idx],
            'delta': self.deltas[idx]
        }

    def get_item_extended(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            'seq': self.sequences[idx],
            'ohe_seq': self.ohe_sequences[idx],
            'cell_type': self.cell_types[idx],
            'expression': self.expressions[idx],
            'delta': self.deltas[idx]
        }

    def __len__(self):
        return len(self.sequences)


def preprocess_df(
        utr_type: int,
        df: pd.DataFrame,
        mean_expr: float | None = None,
) -> tuple[
    list[str],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor]
]:
    """
    Precalculates OneHot-encoded RNA sequences and prepares necessary
    sequence metadata for conditioning.
    :param utr_type: either 5 or 3, corresponding to 5' or 3' untranslated region.

    :param df: pandas dataframe with the following columns:
     `seq,cell_type,fold,1,2,3,4,mass_center,mass_center_mean,diff,zscore,mass_center_std`

    :param mean_expr: mean expression for deltas calculation. If not passed,
     then the mean expression is calculated on the basis of expressions present in dataset

    :return: tuple of lists. The first list correspond to the sequences,
     the second one is their one-hot encoded sequences versions,
     others contain preprocessed sequence metadata like cell type and expression.
    """
    sequences = []
    ohe_sequences = []
    cell_types = []
    expressions = []

    assert utr_type in [5, 3]

    num_cell_types = 5 if utr_type == 5 else 7
    cell_type2idx = dict(zip(idx2cell_type, range(0, num_cell_types)))

    ohe = OneHotSeqEncoder()

    for seq, group in tqdm(df.groupby('seq'), desc='Loading UTR data...'):
        group_exprs = torch.Tensor(np.array(group[['1', '2', '3', '4']]))
        group_exprs /= group_exprs.sum(dim=1).view(-1, 1)

        sequences.extend([seq] * len(group))
        ohe_sequences.extend([ohe(seq)] * len(group))
        cell_types.extend(torch.tensor([cell_type2idx[cell_type]])
                          for cell_type in group['cell_type'])
        expressions.extend((group_exprs @ torch.Tensor([1, 2, 3, 4]))[None])

    ohe_sequences = torch.cat([seq[None] for seq in ohe_sequences])
    cell_types = torch.cat(cell_types)
    expressions = torch.cat(expressions)

    if mean_expr is None:
        return (
            sequences,
            ohe_sequences,
            cell_types,
            expressions,
            expressions - expressions.mean(),
            expressions.mean()
        )

    return (
        sequences,
        ohe_sequences,
        cell_types,
        expressions,
        expressions - mean_expr
    )


def create_datasets(
        utr_type: int,
        table_path: str | Path,
) -> tuple[UTRDataset, ...]:
    df = pd.read_csv(table_path)
    train_df = df[df['fold'] == 'train']
    val_df = df[df['fold'] == 'val']
    test_df = df[df['fold'] == 'test']

    train_preprocessed = preprocess_df(utr_type, train_df)
    mean_expr = train_preprocessed[-1]
    train_dataset = UTRDataset(*train_preprocessed[:-1])
    val_dataset = UTRDataset(*preprocess_df(utr_type, val_df, mean_expr))
    test_dataset = UTRDataset(*preprocess_df(utr_type, test_df, mean_expr))
    return train_dataset, val_dataset, test_dataset


def collate_dict_batch(
    batch_list: list[dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    keys = batch_list[0].keys()
    output_batch = {}

    for key in keys:
        output_batch[key] = torch.cat(
            [dct[key][None] for dct in batch_list],
            dim=0
         )
    return output_batch
