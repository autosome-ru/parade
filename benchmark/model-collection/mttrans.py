
import torch
from torch import nn
# import torch.nn.functional as F
from scipy import stats

import numpy as np


# --------------------------------------------------------
# -------------------- ENCODER LAYERS --------------------
# --------------------------------------------------------

class Conv1d_block(nn.Module):
    """
    the Convolution backbone define by a list of convolution block
    """

    def __init__(self, channel_ls, kernel_size, stride, padding_ls=None, diliation_ls=None, pad_to=None, activation='ReLU'):
        """
        Argument
            channel_ls : list, [int], channel for each conv layer
            kernel_size : int
            stride :  list, [int]
            padding_ls :   list, [int]
            diliation_ls : list, [int]
        """
        super(Conv1d_block, self).__init__()
        # property
        self.activation = activation
        self.channel_ls = channel_ls
        self.kernel_size = kernel_size
        self.stride = stride
        if padding_ls is None:
            self.padding_ls = [0] * (len(channel_ls) - 1)
        else:
            assert len(padding_ls) == len(channel_ls) - 1
            self.padding_ls = padding_ls
        if diliation_ls is None:
            self.diliation_ls = [1] * (len(channel_ls) - 1)
        else:
            assert len(diliation_ls) == len(channel_ls) - 1
            self.diliation_ls = diliation_ls

        self.encoder = nn.ModuleList(
            #                   in_C         out_C           padding            diliation
            [self.Conv_block(channel_ls[i], channel_ls[i+1], self.padding_ls[i], self.diliation_ls[i], self.stride[i]) for i in range(len(self.padding_ls))]
        )

    def Conv_block(self, in_Chan, out_Chan, padding, dilation, stride): 

        activation_layer = eval(f"nn.{self.activation}")

        block = nn.Sequential(
                nn.Conv1d(in_Chan, out_Chan, self.kernel_size, stride, padding, dilation),
                nn.BatchNorm1d(out_Chan),
                activation_layer())

        return block

    def forward(self, x):
        if x.shape[2] == 4:
            out = x.transpose(1, 2)
        else:
            out = x
        for block in self.encoder:
            out = block(out)
        return out

    def forward_stage(self, x, stage):
        """
        return the activation of each stage for exchanging information
        """
        assert stage < len(self.encoder)

        out = self.encoder[stage](x)
        return out

    def cal_out_shape(self, L_in=100, padding=0, diliation=1, stride=2):
        """
        For convolution 1D encoding , compute the final length 
        """
        L_out = 1 + (L_in + 2*padding - diliation*(self.kernel_size-1) - 1) / stride
        return L_out

    def last_out_len(self, L_in=100):
        for i in range(len(self.padding_ls)):
            padding = self.padding_ls[i]
            diliation = self.diliation_ls[i]
            stride = self.stride[i]
            L_in = self.cal_out_shape(L_in, padding, diliation, stride)
        # assert int(L_in) == L_in , "convolution out shape is not int"

        return int(L_in) if L_in >= 0 else 1


class ConvTranspose1d_block(Conv1d_block):
    """
    the Convolution transpose backbone define by a list of convolution block
    """

    def __init__(self, channel_ls, kernel_size, stride, padding_ls=None, diliation_ls=None, pad_to=None):
        channel_ls = channel_ls[::-1]
        stride = stride[::-1]
        padding_ls = padding_ls[::-1] if padding_ls is not None else [0] * (len(channel_ls) - 1)
        diliation_ls = diliation_ls[::-1] if diliation_ls is not None else [1] * (len(channel_ls) - 1)
        super(ConvTranspose1d_block, self).__init__(channel_ls, kernel_size, stride, padding_ls, diliation_ls, pad_to)

    def Conv_block(self, in_Chan, out_Chan, padding, dilation, stride): 
        """
        replace `Conv1d` with `ConvTranspose1d`
        """
        block = nn.Sequential(
                    nn.ConvTranspose1d(in_Chan, out_Chan, self.kernel_size, stride, padding, dilation=dilation),
                    nn.BatchNorm1d(out_Chan),
                    nn.ReLU()
                )

        return block

    def cal_out_shape(self, L_in, padding=0, diliation=1, stride=1, out_padding=0):
        #                  L_in=100,padding=0,diliation=1,stride=2
        """
        For convolution Transpose 1D decoding, compute the final length
        """
        L_out = (L_in - 1) * stride + diliation * (self.kernel_size - 1) + 1 - 2 * padding + out_padding 
        return L_out


class linear_block(nn.Module):
    def __init__(self, in_Chan, out_Chan, dropout_rate=0.2):
        """
        building block func to define dose network
        """
        super(linear_block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_Chan, out_Chan),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(out_Chan),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


# ------------------------------------------------------
# -------------------- MODEL LAYERS --------------------
# ------------------------------------------------------

class backbone_model(nn.Module):
    def __init__(self, conv_args, activation='ReLU'):
        """
        the most bottle model which define a soft-sharing convolution block some forward method
        """
        super(backbone_model, self).__init__()
        channel_ls, kernel_size, stride, padding_ls, diliation_ls, pad_to = conv_args

        self.channel_ls = channel_ls
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_ls = padding_ls
        self.diliation_ls = diliation_ls
        self.pad_to = pad_to

        # model
        self.soft_share = Conv1d_block(channel_ls, kernel_size, stride, padding_ls, diliation_ls, activation=activation)
        # property
        self.stage = list(range(len(channel_ls)-1))
        self.out_length = self.soft_share.last_out_len(pad_to)
        self.out_dim = self.soft_share.last_out_len(pad_to)*channel_ls[-1]

    def _weight_initialize(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias)
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)
        elif isinstance(model, nn.Conv1d):
            nn.init.orthogonal_(model.weight)
        elif isinstance(model, nn.Conv2d):
            nn.init.kaiming_normal_(model.weight, nonlinearity='leaky_relu',)
        elif isinstance(model, nn.BatchNorm1d):
            nn.init.constant_(model.weight, 1)
            nn.init.constant_(model.bias, 0)

    def forward_stage(self, X, stage):
        return self.soft_share.forward_stage(X, stage)

    def forward_tower(self, Z):
        """
        Each new backbone model should re-write the `forward_tower` method
        """
        return Z

    def forward(self, X):
        Z = self.soft_share(X)
        out = self.forward_tower(Z)
        return out


class RL_regressor(backbone_model):

    def __init__(self, conv_args, tower_width=40, dropout_rate=0.2, activation='ReLU'):
        """
        backbone for RL regressor task  ,the same soft share should be used among task
        Arguments:
            conv_args: (channel_ls,kernel_size,stride,padding_ls,diliation_ls)
        """
        super(RL_regressor, self).__init__(conv_args, activation)

        # ------- architecture --------
        # self.tower = linear_block(in_Chan=self.out_dim,out_Chan=tower_width,dropout_rate=dropout_rate)
        # self.fc_out = nn.Linear(tower_width,1)

        # ------- task specific -------
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.task_name = 'RL_regression'
        self.loss_dict_keys = ['Total']

    def forward_tower(self, Z):
        # flatten
        batch_size = Z.shape[0]
        Z_flat = Z.view(batch_size, -1)
        # tower part
        Z_to_out = self.tower(Z_flat)
        out = self.fc_out(Z_to_out)
        return out

    def squeeze_out_Y(self, out, Y):
        # ------ squeeze ------
        if len(Y.shape) == 2:
            Y = Y.squeeze(1)
        if len(out.shape) == 2:
            out = out.squeeze(1)

        assert Y.shape == out.shape, "keep label and pred the same shape"
        return out, Y

    def compute_acc(self, out, X, Y, popen=None):
        # try:
        #     epsilon = popen.epsilon
        # except:
        #     epsilon = 0.3

        out, Y = self.squeeze_out_Y(out, Y)
        # error smaller than epsilon
        with torch.no_grad():
            y_ay = Y.cpu().numpy()
            out_ay = out.cpu().numpy()
            # acc = torch.sum(torch.abs(Y-out) < epsilon).item() / Y.shape[0]
            acc = stats.spearmanr(y_ay, out_ay)[0]
            # acc = r2_score(y_ay, out_ay)
        return {"Acc": acc}

    def compute_loss(self, out, X, Y, popen):
        out, Y = self.squeeze_out_Y(out, Y)
        loss = self.loss_fn(out, Y) + popen.l1 * torch.sum(torch.abs(next(self.soft_share.encoder[0].parameters())))
        return {"Total": loss}


class RL_gru(RL_regressor):
    def __init__(self, conv_args, tower_width=40, dropout_rate=0.2, activation='ReLU', in_channels=4):
        """
        tower is gru
        """
        # adding 'in_channels' to conv_args[0]
        super().__init__(([in_channels] + conv_args[0],) + conv_args[1:], tower_width, dropout_rate, activation)
        self.configure_towerwidth(tower_width)
        # previous, it is a linear layer
        if dropout_rate > 0:
            self.soft_share.encoder = nn.ModuleList([
                nn.Sequential(conv_layer, nn.Dropout(dropout_rate))
                for conv_layer in self.soft_share.encoder
             ])
        self.tower = nn.GRU(input_size=self.channel_ls[-1],
                            hidden_size=self.tower_width,
                            num_layers=2,
                            batch_first=True)  # input: batch, seq, features
        self.fc_out = nn.Linear(self.tower_width, 1)

        self.apply(self._weight_initialize)

    def configure_towerwidth(self, tower_width):
        if isinstance(tower_width, int):
            self.tower_width = tower_width
        elif isinstance(tower_width, list):
            self.tower_width = tower_width[0]
        elif isinstance(tower_width, dict):
            self.tower_width = tower_width.values()[0]

    def forward_tower(self, Z):
        # flatten
        # batch_size = Z.shape[0]
        Z_flat = torch.transpose(Z, 1, 2)
        # tower part
        h_prim, (c1, c2) = self.tower(Z_flat)  # [B,L,h] , [B,h] cell of layer 1, [B,h] of layer 2
        out = self.fc_out(c2)
        return out

    @torch.no_grad()
    def predict_each_position(self, X):
        Z = self.soft_share(X)
        Z_flat = torch.transpose(Z, 1, 2)
        # tower part
        h_prim, (c1, c2) = self.tower(Z_flat)  # [B,L,h] , [B,h] cell of layer 1, [B,h] of layer 2 
        out_series = self.fc_out(h_prim)
        return out_series


# -------------------------------------------------------
# ----------------- OPTIMIZER/SCHEDULER -----------------
# -------------------------------------------------------

from torch.optim.lr_scheduler import _LRScheduler

# The 'ScheduledOptimizer' in the original code is actually Adam with NoamLR Scheduler
class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


def get_optimizer(lr=0.0003, l2=1e-06):
    return dict(
        optimizer_class=torch.optim.Adam,
        optimizer_kws=dict(
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-09,
            weight_decay=l2,
            amsgrad=True,
        )
    )


def get_scheduler():
    return dict(
        lr_scheduler_class=NoamLR,
        lr_scheduler_kws=dict(
            warmup_steps=20,
        ),
    )