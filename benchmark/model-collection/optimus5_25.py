from collections import OrderedDict

import torch
from torch import nn
# import torch.nn.functional as F


class Optimus5Prime25(nn.Module):
    def __init__(
        self,
        seqsize,  # e.g., 25nt
        in_channels=6,
        out_channels=1,
        conv_layers=2,
        conv_kernel_size=5,
        conv_filters_first=128,
        conv_dropout=0.45,
        dense_latent=18,
        dense_dropout=0.0,
        final_activation=nn.Identity,
        # reg_lambda=0.0,
    ):
        super().__init__()

        self.convs = nn.Sequential()

        channel_sizes = [in_channels] + [conv_filters_first * (2**i) for i in range(conv_layers)]
        seqsize_transf = seqsize

        for in_channels, step_channels in zip(channel_sizes[:-1], channel_sizes[1:]):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=step_channels,
                        kernel_size=conv_kernel_size,
                        padding='same',
                        # kernel_regularizer is L2. In Torch, it gets replaced by w-t decay (e.g. Adam -> AdamW)
                        # however, `reg_lambda` is always 0.0 in the original code, so it is probably unimportant
                    ),
                    nn.ReLU(),
                    nn.Conv1d(
                        in_channels=step_channels,  # stacked upon 1st
                        out_channels=step_channels,
                        kernel_size=conv_kernel_size,
                        padding='same',
                    ),
                    nn.ReLU(),
                    nn.MaxPool1d(
                        kernel_size=2,
                        stride=2,
                        padding=0,
                    ),
                    nn.Dropout(
                        p=conv_dropout,
                    ),
                )
            )
            seqsize_transf //= 2

        self.convs.append(nn.Flatten())

        self.linear = nn.Sequential(
            nn.Linear(seqsize_transf * channel_sizes[-1], dense_latent),
            nn.ReLU(),
            nn.Dropout(p=dense_dropout),
            nn.Linear(dense_latent, out_channels),
            final_activation()
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.linear(x)
        return x


def get_optimizer():
    return dict(
        optimizer_class=torch.optim.Adam,
        optimizer_kws=dict(lr=0.001)
    )


class lr_lambda(object):
    def __init__(self, decay_rate, decay_steps):
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def __call__(self, step):
        return 1.0 / (1.0 + self.decay_rate * step / self.decay_steps)


def get_scheduler(steps_per_epoch):
    # Equivalent to tf.keras.optimizers.schedules.InverseTimeDecay
    dec_rate = 0.001

    return dict(
        lr_scheduler_class=torch.optim.lr_scheduler.MultiplicativeLR,
        lr_scheduler_kws=dict(
            lr_lambda=lr_lambda(dec_rate, steps_per_epoch * 10)  # twice per 10 epochs
        )
    )