import torch
from torch import nn

from prixfixe.autosome import AutosomeFirstLayersBlock
from prixfixe.bhi import BHIFirstLayersBlock
from prixfixe.bhi import BHICoreBlock
from prixfixe.unlockdna import UnlockDNACoreBlock
from prixfixe.prixfixe import FinalLayersBlock
# from prixfixe.autosome import AutosomeFinalLayersBlock
from prixfixe.prixfixe import PrixFixeNet


class AdaptedAutosomeFinalLayersBlock(FinalLayersBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seqsize=None,
        mapper_size=256,
        linear_sizes=None,
        use_max_pooling=False,
        final_activation=nn.Identity,
        bn_momentum=0.1
    ):
        super().__init__(in_channels=in_channels,
                         seqsize=seqsize)

        self.mapper = block = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=mapper_size,
                kernel_size=1,
                padding='same',
            ),
            # activation()
        )

        self.avgpooling = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        if linear_sizes is not None:
            first_linear = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(mapper_size, linear_sizes[0]),
                nn.BatchNorm1d(linear_sizes[0],
                               momentum=bn_momentum),
                nn.SiLU()
            )

            linear_blocks = list()
            for prev_sz, sz in zip(linear_sizes[:-1], linear_sizes[1:]):
                block = nn.Sequential(
                    nn.Dropout(0.1),
                    nn.Linear(prev_sz, sz),
                    nn.BatchNorm1d(sz,
                                   momentum=bn_momentum),
                    nn.SiLU()
                )
                linear_blocks.append(block)

            last_linear = nn.Sequential(
                nn.Linear(linear_sizes[-1], out_channels),
                final_activation()
            )

            self.linear = nn.Sequential(
                nn.Dropout(0.1),
                first_linear,
                *linear_blocks,
                last_linear,
            )
        else:  # i.e. if self.linear_sizes is None
            self.linear = nn.Sequential(
                nn.Linear(mapper_size, out_channels),
                final_activation()
            )

    def forward(self, x):
        x = self.mapper(x)
        x = self.avgpooling(x)
        x = self.linear(x)
        return x

    def train_step(self, *args, **kwargs):
        raise NotImplementedError


class DreamRNNFacade(nn.Module):
    def __init__(
        self,
        seqsize,  # e.g., 25nt
        in_channels=6,
        out_channels=1,
    ):
        super().__init__()

        self.first = BHIFirstLayersBlock(
            in_channels=in_channels,
            out_channels=320,
            seqsize=seqsize,
            kernel_sizes=(9, 15),
            pool_size=1,
            dropout=0.2,
        )

        self.core = BHICoreBlock(
            in_channels=self.first.out_channels,
            out_channels=320,
            seqsize=self.first.infer_outseqsize(),
            lstm_hidden_channels=320,
            kernel_sizes=[9, 15],
            pool_size=1,
            dropout1=0.2,
            dropout2=0.5,
        )

        self.final = AdaptedAutosomeFinalLayersBlock(
            in_channels=self.core.out_channels,
            out_channels=out_channels,
            mapper_size=256,
            linear_sizes=None,
            use_max_pooling=False,
            final_activation=nn.Identity,
            bn_momentum=0.1
        )

        self.model = PrixFixeNet(
            first=self.first,
            core=self.core,
            final=self.final,
            generator=torch.Generator()
        )

    def forward(self, x):
        return self.model(x)


class DreamAttnFacade(nn.Module):
    def __init__(
        self,
        seqsize,  # e.g., 25nt
        in_channels=6,
        out_channels=1,
    ):
        super().__init__()

        self.first = AutosomeFirstLayersBlock(
            in_channels=in_channels,
            out_channels=256,
            seqsize=seqsize,
        )

        self.core = UnlockDNACoreBlock(
            in_channels=self.first.out_channels,
            out_channels=self.first.out_channels,
            seqsize=seqsize,
            n_blocks=4,
            kernel_size=15,
            rate=0.1,
            num_heads=8,
        )

        self.final = AdaptedAutosomeFinalLayersBlock(
            in_channels=self.core.out_channels,
            out_channels=out_channels,
            mapper_size=256,
            linear_sizes=None,
            use_max_pooling=False,
            final_activation=nn.Identity,
            bn_momentum=0.1
        )

        self.model = PrixFixeNet(
            first=self.first,
            core=self.core,
            final=self.final,
            generator=torch.Generator()
        )

    def forward(self, x):
        return self.model(x)


def get_optimizer(
    lr=0.001,
    div_factor=25.0,
    weight_decay=0.01,
):

    return dict(
        optimizer_class=torch.optim.AdamW,
        optimizer_kws=dict(
            lr=lr / div_factor,
            weight_decay=weight_decay
        )
    )


def get_scheduler(
    epochs,
    steps_per_epoch,
    lr=0.001,
    div_factor=25.0,
):
    return dict(
        lr_scheduler_class=torch.optim.lr_scheduler.OneCycleLR,
        lr_scheduler_kws=dict(
            max_lr=lr,
            div_factor=div_factor,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.3,
            three_phase=False,
        )
    )
