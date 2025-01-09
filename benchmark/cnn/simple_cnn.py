from collections import OrderedDict

import torch
from torch import nn
# import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(
        self,
        seqsize,
        in_channels=6,
        out_channels=1,
        conv_sizes=(120, 120, 120),
        dropouts=(0, 0, 0.2),
        linear_size=40,
        ks=8,
        activation=nn.ReLU,
        final_activation=nn.Identity
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_sizes = conv_sizes
        self.dropouts = dropouts
        self.linear_size = linear_size
        self.seqsize = seqsize

        self.convs = nn.Sequential()

        if len(self.conv_sizes) >= 1:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=self.in_channels,
                        out_channels=self.conv_sizes[0],
                        kernel_size=ks,
                        padding='same'
                    ),
                    activation()
                )
            )

        if len(self.conv_sizes) >= 2:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=self.conv_sizes[0],
                        out_channels=self.conv_sizes[1],
                        kernel_size=ks,
                        padding='same'
                    ),
                    activation(),
                    nn.Dropout(p=self.dropouts[0])
                )
            )

        if len(self.conv_sizes) >= 3:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=self.conv_sizes[1],
                        out_channels=self.conv_sizes[2],
                        kernel_size=ks,
                        padding='same'
                    ),
                    activation(),
                    nn.Dropout(p=self.dropouts[1])
                )
            )

        if len(self.conv_sizes) >= 4 or len(self.conv_sizes) < 1:
            raise NotImplementedError("Only 1-3 conv1d layers are supported")

        self.convs.append(nn.Flatten())

        self.linear = nn.Sequential(
            nn.Linear(self.seqsize * self.conv_sizes[-1], self.linear_size),
            activation(),
            nn.Dropout(p=self.dropouts[-1]),
            nn.Linear(self.linear_size, self.out_channels),
            final_activation()
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.linear(x)
        return x