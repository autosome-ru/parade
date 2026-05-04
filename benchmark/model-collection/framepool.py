from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F


class ConvolveBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding='same',
        dropout=0.0,
        residual=True,
    ):
        super().__init__()
        self.residual = residual
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding
            ),
            nn.ReLU(),
            nn.Dropout1d(p=dropout),
        )

    def forward(self, x):
        x_tr = self.block(x)
        if self.residual:
            x_tr = x_tr + x
        return x_tr


class FramePool(nn.Module):
    def __init__(
        self,
        in_channels=6,
        conv_channels=128,
        out_channels=1,
        n_conv_layers=3,
        kernel_sizes=(7, 7, 7),
        dilations=(1, 1, 1),
        dropouts=(0.0, 0.0, 0.0),
        padding='same',
        skip_connections=True,
        n_fc_layers=1,
        fc_layer_sizes=(64,),
        fc_dropouts=(0.2,),
        only_max_pool=False,
    ):
        super().__init__()

        self.only_max_pool = only_max_pool

        self.convs = nn.Sequential()

        for i, layer_ks, layer_dilation, layer_dropout \
                in zip(range(n_conv_layers), kernel_sizes, dilations, dropouts):
            if i == 0:
                layer_residual = False
                layer_in_channels = in_channels
            else:
                layer_residual = skip_connections
                layer_in_channels = conv_channels
            self.convs.append(
                ConvolveBlock(
                    in_channels=layer_in_channels,
                    out_channels=conv_channels,  # doesn't change
                    kernel_size=layer_ks,
                    padding=padding,
                    dropout=layer_dropout,
                    residual=layer_residual,
                )
            )

        # Blockless pooling-reshaping procedures; defined in 'forward'

        self.fc = nn.Sequential(nn.Flatten())

        fc_input_size = conv_channels * 3 * (2 - self.only_max_pool)  # 3 or 6 values per channel
        adj_fc_layer_sizes = (fc_input_size, *fc_layer_sizes)
        for i, layer_input, layer_output, layer_dropout \
                in zip(range(n_fc_layers), adj_fc_layer_sizes[:-1], adj_fc_layer_sizes[1:], fc_dropouts):
            self.fc.append(
                nn.Sequential(
                    nn.Linear(layer_input, layer_output),
                    nn.ReLU(),
                    nn.Dropout(layer_dropout)
                )
            )
        self.fc.append(
            nn.Linear(adj_fc_layer_sizes[-1], out_channels)
        )

    def forward(self, x):
        x = self.convs(x)
        x = torch.flip(x, dims=(-1,))
        frame_1 = x[..., 0::3]
        frame_2 = x[..., 1::3]
        frame_3 = x[..., 2::3]
        if self.only_max_pool:
            frame_pools = torch.cat(
                [
                    F.adaptive_max_pool1d(frame_1, 1),
                    F.adaptive_max_pool1d(frame_2, 1),
                    F.adaptive_max_pool1d(frame_3, 1),
                ],
                dim=-1,
            )
        else:
            frame_pools = torch.cat(
                [
                    F.adaptive_max_pool1d(frame_1, 1),
                    F.adaptive_avg_pool1d(frame_1, 1),
                    F.adaptive_max_pool1d(frame_2, 1),
                    F.adaptive_avg_pool1d(frame_2, 1),
                    F.adaptive_max_pool1d(frame_3, 1),
                    F.adaptive_avg_pool1d(frame_3, 1),
                ],
                dim=-1,
            )
        x = self.fc(frame_pools)
        return x


def get_optimizer():
    return dict(
        optimizer_class=torch.optim.Adam,
        optimizer_kws=dict(
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    )


def get_scheduler():
    return dict(
        lr_scheduler_class=None,
        lr_scheduler_kws=dict(),
    )