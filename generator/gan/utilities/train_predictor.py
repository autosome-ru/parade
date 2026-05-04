import sys

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


sys.path.append("../model/")
import utilities.utrdata_cl as utrdata # type: ignore
from utilities.pl_regressor import RNARegressor



def launch_model(
    seed: int,
    data,
    cuda,
    batch_size: int,
    num_workers: int,
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
    exp_type: str,
):
    pl.seed_everything(seed)

    # Creating Datasets
    train_set = utrdata.UTRData(
        df=data["train"],
        **train_ds_kws,
    )
    val_set = utrdata.UTRData(
        df=data["val"],
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
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'{exp_type}_seed_{seed}',
        save_top_k=1,
        save_last=False,
        monitor="val_pearson_r_0",
        mode="max"
    )
    progressbar_callback = pl.callbacks.TQDMProgressBar(refresh_rate=0.5)

    logger = pl.loggers.tensorboard.TensorBoardLogger("tb_logs", name=model.model_name)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, progressbar_callback],
        logger=logger,
        accelerator="gpu",
        devices=cuda,
        deterministic=True,
        max_epochs=epochs,
        num_sanity_val_steps=0,
        # gradient_clip_val=1e-5,
        # gradient_clip_algorithm="norm",
    )
    trainer.fit(model=model, train_dataloaders=dl_train, val_dataloaders=dl_val)
    best_model = RNARegressor.load_from_checkpoint(checkpoint_callback.best_model_path)
    print(checkpoint_callback.best_model_path)
    prediction = trainer.predict(model=best_model, dataloaders=dl_val)
    val_pred, val_real = zip(*prediction)
    val_pred = torch.concat(val_pred).numpy()
    val_real = torch.concat(val_real).numpy()
    val_df = data["val"].copy()
    val_df["real_0"] = val_real[:, 0]
    val_df["real_1"] = val_real[:, 1]
    val_df["pred_0"] = val_pred[:, 0]
    val_df["pred_1"] = val_pred[:, 1]

    return trainer, val_df, checkpoint_callback.best_model_path
