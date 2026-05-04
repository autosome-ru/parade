import os
import sys

import itertools

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
addpath = os.path.normpath(os.path.join(__location__, "./"))
if addpath not in sys.path:
    sys.path.append(addpath)

import utrdata_cl as utrdata
import stability_data
from legnet import LegNetClassifier
from pl_regressor import RNARegressor


CELLTYPE_CODES_UTR5 = ['c1', 'c2', 'c4', 'c6', 'c17']
CELLTYPE_CODES_UTR3 = ['c1', 'c2', 'c4', 'c6', 'c17', 'c13']

MODEL_PATH_UTR5 = os.path.normpath(os.path.join(__location__, "../regression_multiple/saved_models/model-utr5-deltas-epoch=9-step=840.ckpt"))
MODEL_PATH_UTR3 = os.path.normpath(os.path.join(__location__, "../regression_multiple/saved_models/model-utr3-deltas-epoch=9-step=1330.ckpt"))
MODEL_PATH_STABILITY = os.path.normpath(os.path.join(__location__, "../regression_stability/saved_models/stability-epoch=24-step=725.ckpt"))


def create_dataset_activity(seqs: list[str], cell_line_codes: list[str]) -> pd.DataFrame:
    col_idseq, col_cell_type = zip(*itertools.product(enumerate(seqs), cell_line_codes))
    col_id, col_seq = zip(*col_idseq)
    df = pd.DataFrame({'id': col_id, 'seq': col_seq, 'cell_type': col_cell_type})
    return df

def create_dataset_stability(seqs: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({'id': np.arange(len(seqs)), 'seq': seqs})
    return df


def predict_sequences(
    seqs: list[str],
    utr_type: str,
    cell_line_codes=None,
    model_path=None,
    device='cuda:0',
    batch_size=1024,
    num_workers=32,
) -> pd.DataFrame:
    if model_path is None:
        if utr_type == "utr5":
            model_path = MODEL_PATH_UTR5
        elif utr_type == "utr3":
            model_path = MODEL_PATH_UTR3
        elif utr_type == "stability":
            model_path = MODEL_PATH_STABILITY
        else:
            raise ValueError

    if cell_line_codes is None:
        if utr_type == "utr5":
            cell_line_codes = list(CELLTYPE_CODES_UTR5)
        elif utr_type == "utr3":
            cell_line_codes = list(CELLTYPE_CODES_UTR3)
        elif utr_type == "stability":
            pass
        else:
            raise ValueError

    if utr_type.startswith("utr"):
        ds_frame = create_dataset_activity(seqs, cell_line_codes)
        dl = DataLoader(
            utrdata.UTRData(
                df=ds_frame,
                predict_cols=[],
                construct_type=utr_type,
                features=("sequence", "positional", "conditions"),
                augment=False,
                augment_test_time=False,
                augment_kws=dict(
                    extend_left=0,
                    extend_right=0,
                    shift_left=0,
                    shift_right=0,
                    revcomp=False,
                ),
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )
    elif utr_type == "stability":
        ds_frame = create_dataset_stability(seqs)
        dl = DataLoader(
            stability_data.StabilityData(
                df=ds_frame,
                features=("sequence",),
                predict_cols=[],
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False,
        )

    model = RNARegressor.load_from_checkpoint(model_path, weights_only=False)
    trainer = pl.Trainer(
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=0.5)],
        logger=False,
        accelerator="gpu",
        devices=[0],
        deterministic=True,
        num_sanity_val_steps=0,
    )
    output = trainer.predict(model=model, dataloaders=dl)
    pred, _ = zip(*output)
    pred = torch.concat(pred).numpy()
    if utr_type.startswith("utr"):
        ds_frame["pred_mass_center"] = pred[:, 1]
        pivot_frame = ds_frame.pivot(index=["id", "seq"], columns="cell_type", values="pred_mass_center").reset_index(level=0, drop=True)
        return pivot_frame
    elif utr_type == "stability":
        ds_frame["pred_log_ratio"] = pred[:, 0]
        return ds_frame.drop("id", axis=1).set_index("seq")["pred_log_ratio"]