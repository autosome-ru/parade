import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as ss

import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
model_module_path = os.path.normpath(os.path.join(__location__, "../../model"))
sys.path.append(model_module_path)

import utrdata_cl as utrdata
from legnet import LegNetClassifier
from pl_regressor import RNARegressor


def load_data(utr_type, prefix="."):
    if utr_type.lower() == "utr3":
        PATH_FROM = os.path.join(prefix, "UTR3_zscores_replicateagg.csv")
    elif utr_type.lower() == "utr5":
        PATH_FROM = os.path.join(prefix, "UTR5_zscores_replicateagg.csv")
    df = pd.read_csv(PATH_FROM)

    splits = dict(tuple(df.groupby('fold')))
    for split_df in splits.values():
        split_df.reset_index(drop=True, inplace=True)
    return splits


def launch_model(
    model_name: str,
    utr_type: str,
    splits: dict,
    seed: int,
    batch_size: int,
    train_ds_kws: dict,
    val_ds_kws: dict,
    model_class,
    model_kws: dict,
    criterion_class,
    criterion_kws: dict,
    optimizer_class,
    optimizer_kws: dict,
    lr_scheduler_class,
    lr_scheduler_kws: dict,
    test_time_validation: bool,
    epochs: int,
    num_workers: int,
    device: int = None,
):
    pl.seed_everything(seed)

    # Creating Datasets
    train_set = utrdata.UTRData(
        df=splits["train"],
        **train_ds_kws,
    )
    val_set = utrdata.UTRData(
        df=splits["val"],
        **val_ds_kws,
    )
    test_set = utrdata.UTRData(
        df=splits["test"],
        **val_ds_kws,
    )

    assert train_set.num_channels == val_set.num_channels
    try:
        div_factor = val_ds_kws["augment_kws"]["shift_left"] + \
                     val_ds_kws["augment_kws"]["shift_right"] + 1
    except KeyError:
        div_factor = 1

    # Creating DataLoaders
    dl_train = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True
    )
    # dl_train = utrdata.DataLoaderWrapper(dl_train, batch_per_epoch=batch_per_epoch)
    dl_val = DataLoader(
        val_set,
        batch_size=batch_size // div_factor,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False
    )
    dl_test = DataLoader(
        test_set,
        batch_size=batch_size // div_factor,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False
    )

    model = RNARegressor(
        model_class=model_class,
        model_kws=model_kws | dict(
            in_channels=train_set.num_channels
        ),
        criterion_class=criterion_class,
        criterion_kws=criterion_kws,
        optimizer_class=optimizer_class,
        optimizer_kws=optimizer_kws,
        lr_scheduler_class=lr_scheduler_class,
        lr_scheduler_kws=lr_scheduler_kws,
        test_time_validation=test_time_validation,
    )
    model.model_name = model_name

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"saved_models/{utr_type}_{model_name}/seed={seed:02d}/",
        save_top_k=1,
        save_last=False,
        monitor=f"val_pearson_r_{len(val_ds_kws['predict_cols']) - 1}",
        mode="max"
    )
    progressbar_callback = pl.callbacks.TQDMProgressBar(refresh_rate=0.5)
    learningrate_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    logger = pl.loggers.tensorboard.TensorBoardLogger("tb_logs", name=model.model_name)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, progressbar_callback, learningrate_callback],
        logger=logger,
        accelerator="gpu",
        devices=1 if device is None else [device],
        deterministic=True,
        max_epochs=epochs,
        num_sanity_val_steps=0,
        # gradient_clip_val=1e-5,
        # gradient_clip_algorithm="norm",
    )
    trainer.fit(model=model, train_dataloaders=dl_train, val_dataloaders=dl_val)
    best_model = RNARegressor.load_from_checkpoint(checkpoint_callback.best_model_path)

    prediction = trainer.predict(model=best_model, dataloaders=dl_test)
    test_pred, test_real = zip(*prediction)
    test_pred = torch.concat(test_pred).numpy()
    test_real = torch.concat(test_real).numpy()
    test_mass_center_pred = test_pred[:, -1]  # Last (or the only) column should contain the predicted mass center
    test_mass_center_real = test_real[:, -1]
    test_df = splits["test"].copy()
    assert np.allclose(test_df["mass_center"].values, test_mass_center_real)
    test_df["pred_mass_center"] = test_mass_center_pred

    metrics_ct = list()
    cell_types = ["all"]
    cell_types.extend(sorted(test_df["cell_type"].unique()))
    for ct in cell_types:
        if ct == "all":
            grouping = test_df.groupby("seq")
            real = grouping["mass_center"].mean()
            pred = grouping["pred_mass_center"].mean()
        else:
            ct_filter = test_df["cell_type"] == ct
            real = test_df.loc[ct_filter, "mass_center"]
            pred = test_df.loc[ct_filter, "pred_mass_center"]
        r = ss.pearsonr(pred, real)
        rho = ss.spearmanr(pred, real)
        metrics = {
            "model": model_name,
            "cell type": ct,
            "seed": seed,
            "pearsonr": r.statistic,
            "pearsonr_pvalue": r.pvalue,
            "spearmanr": rho.statistic,
            "spearmanr_pvalue": rho.pvalue,
        }
        metrics_ct.append(metrics)

    return metrics_ct
