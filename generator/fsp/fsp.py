import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fsp_utils import (
    CELLTYPE_CODES_UTR3,
    CELLTYPE_CODES_UTR5,
    gen_random_seqs,
    make_res,
)


######################
##                  ##
## Energy Functions ##
##                  ##
######################


class BaseEnergy(torch.nn.Module):
    """Base class for energy functions used in sequence optimization."""

    def __init__(self):
        super().__init__()
        self.model = None

    def forward(self, x_in):
        return self.energy_calc(x_in)

    def energy_calc(self, x):
        raise NotImplementedError("Energy calculation not implemented.")


class MinGapEnergy(BaseEnergy):
    """
    Energy function based on the gap between target cell activity and
    mean off-target cell activity (replicates the Malinois loss from Gosai et al.).

    The model is expected to return a tensor of shape (batch, 2), where
    index 0 is activity and index 1 is delta. Only index 1 (delta) is used
    here, matching the regressor output convention.

    Args:
        model: Predictor network (returns shape [batch, 2]).
        device: Torch device.
        seq_len: Length of the input sequence.
        utr_type: One of "utr3" or "utr5".
        target_feature: Index of the target cell type in CELLTYPE_CODES.
        target_alpha: Weight for the target cell term.
        bending_factor: Optional value-bending parameter.
        a_min: Lower clamp for model output.
        a_max: Upper clamp for model output.
        loss_type: One of "malinois", "square", "square_adj", "exp".
    """

    def __init__(
        self,
        model,
        device,
        seq_len,
        utr_type,
        target_feature=0,
        target_alpha=1.0,
        bending_factor=0.0,
        a_min=-math.inf,
        a_max=math.inf,
        loss_type="malinois",
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.loss_type = loss_type
        self.utr_type = utr_type
        self.target_feature = target_feature
        self.target_alpha = target_alpha
        self.bending_factor = bending_factor
        self.a_min = a_min
        self.a_max = a_max
        self.device = device
        self.seq_len = seq_len

    def bend(self, x):
        return x - self.bending_factor * (torch.exp(-x) - 1)

    def energy_calc(self, x):
        x = x.to(self.device)

        if self.utr_type == "utr3":
            celltype_codes = CELLTYPE_CODES_UTR3
        else:
            celltype_codes = CELLTYPE_CODES_UTR5

        batch_hook = torch.tensor([]).to(self.device)
        for cell_line in celltype_codes:
            hook = make_res(x, cell_line, seq_len=self.seq_len, utr_type=self.utr_type, device=self.device)
            hook = self.bend(self.model(hook).clamp(self.a_min, self.a_max))
            # index 1 is delta (the optimization target from our regressor)
            hook = hook[:, 1].unsqueeze(dim=0)
            batch_hook = torch.concat([batch_hook, hook])

        # shape: (batch, n_cell_types)
        hook = batch_hook.T
        off_target = [i for i in range(hook.shape[-1]) if i != self.target_feature]

        if self.loss_type == "malinois":
            energy = (
                hook[..., off_target].mean(-1)
                - hook[..., self.target_feature].mul(self.target_alpha)
            )
        elif self.loss_type == "square":
            energy = -(hook[..., off_target].mean(-1) - hook[..., self.target_feature]) ** 2
        elif self.loss_type == "square_adj":
            energy = -(
                (hook[..., off_target] ** 2).sum(-1)
                - (hook[..., self.target_feature] - hook[..., off_target].amax(-1)) ** 2
            )
        elif self.loss_type == "exp":
            energy = (
                torch.exp(hook[..., off_target]).mean(dim=1)
                - torch.exp(hook[..., self.target_feature])
            )

        return energy


def mask_gradients(in_tensor, mask_tensor):
    filter_tensor = 1 - mask_tensor
    grad_pass = in_tensor.mul(filter_tensor)
    grad_block = in_tensor.detach().mul(mask_tensor)
    return grad_pass + grad_block


######################
##                  ##
##    Parameters    ##
##                  ##
######################


class ParamsBase(nn.Module):
    """Base class for sequence parameters."""

    def __init__(self):
        super().__init__()

    @property
    def shape(self):
        return self().shape

    def rebatch(self, input):
        raise NotImplementedError("Rebatch not implemented.")

    def reset(self):
        raise NotImplementedError("Reset not implemented.")


class GumbelSoftmaxParameters(ParamsBase):
    """
    Sequence parameters using Gumbel-Softmax relaxation.

    Args:
        data: Initial parameter tensor.
        batch_dim: Batch dimension index.
        token_dim: Token (nucleotide) dimension index.
        cat_axis: Axis for concatenation in get_logits.
        n_samples: Number of Gumbel samples per forward pass.
        num_classes: Number of nucleotide classes.
        tau: Gumbel-Softmax temperature.
        prior_var: Variance of the prior (unused, kept for API compatibility).
        use_norm: Whether to apply InstanceNorm before sampling.
        use_affine: Whether InstanceNorm uses learnable affine params.
    """

    def __init__(
        self,
        data,
        batch_dim=0,
        token_dim=1,
        cat_axis=-1,
        n_samples=1,
        num_classes=6,
        tau=1.0,
        prior_var=1.0,
        use_norm=False,
        use_affine=False,
    ):
        super().__init__()
        self.register_parameter("theta", nn.Parameter(data.detach().clone()))
        self.cat_axis = cat_axis
        self.batch_dim = batch_dim
        self.token_dim = token_dim
        self.n_samples = n_samples
        self.tau = tau
        self.prior_var = prior_var
        self.use_norm = use_norm
        self.use_affine = use_affine
        self.num_classes = num_classes
        self.n_dims = len(self.theta.shape)
        self.repeater = [-1 for _ in range(self.n_dims)]
        self.batch_size = self.theta.shape[self.batch_dim]

        if self.use_norm:
            self.norm = nn.InstanceNorm1d(num_features=self.num_classes, affine=self.use_affine)
        else:
            self.norm = nn.Identity()

    @property
    def shape(self):
        return self.theta.shape

    def get_logits(self):
        return self.theta

    def get_sample(self, x=None):
        if x is None:
            x = self.theta
        hook = self.norm(x)
        hook = F.gumbel_softmax(
            hook.unsqueeze(0).expand(self.n_samples, *self.repeater),
            tau=self.tau,
            hard=True,
            eps=1e-10,
            dim=self.token_dim + 1 if self.token_dim >= 0 else self.token_dim,
        )
        return hook

    def forward(self, x=None):
        return self.get_sample(x).flatten(0, 1)

    def reset(self):
        self.theta.data = torch.randn_like(self.theta)

    def rebatch(self, input):
        return input.unflatten(0, (self.n_samples, self.batch_size)).mean(dim=0)


######################
##                  ##
##   FastSeqProp    ##
##                  ##
######################


class FastSeqProp(nn.Module):
    """
    Fast SeqProp: gradient-based sequence optimization via Gumbel-Softmax.

    Args:
        energy_fn: Energy function module (lower = better sequence).
        params: Parameter module (e.g. GumbelSoftmaxParameters).
    """

    def __init__(self, energy_fn, params):
        super().__init__()
        self.energy_fn = energy_fn
        self.params = params
        self.energy_fn.eval()

    def run(
        self,
        n_steps=20,
        learning_rate=0.5,
        step_print=10,
        lr_scheduler=True,
        grad_mask=None,
        create_plot=True,
        log_param_hist=False,
    ):
        """
        Run optimization for n_steps and optionally plot the energy trace.

        Returns:
            pd.DataFrame with columns [step, energy] if create_plot is True, else None.
        """
        eta_min = 1e-6 if lr_scheduler else learning_rate

        optimizer = torch.optim.Adam(self.params.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_steps, eta_min=eta_min
        )

        energy_hist = []
        param_hist = []
        pbar = tqdm(range(1, n_steps + 1), desc="Steps", position=0, leave=True)

        for step in pbar:
            optimizer.zero_grad()
            sampled_nucleotides = self.params()
            if grad_mask is not None:
                sampled_nucleotides = mask_gradients(sampled_nucleotides, grad_mask)
            energy = self.energy_fn(sampled_nucleotides)
            energy = self.params.rebatch(energy)
            energy_hist.append(energy.detach().cpu().numpy())
            energy_to_print = energy.mean()
            energy.sum().backward()
            optimizer.step()
            scheduler.step()
            if log_param_hist:
                param_hist.append(np.copy(self.params.theta.detach().cpu().numpy()))
            if step % step_print == 0:
                pbar.set_postfix({"Loss": energy_to_print.item(), "LR": scheduler.get_last_lr()[0]})

        self.energy_hist = np.stack(energy_hist)
        self.param_hist = np.stack(param_hist) if log_param_hist else None

        if create_plot:
            bsz = self.params.theta.shape[0]
            plot_data = pd.DataFrame({
                "step": np.repeat(np.arange(n_steps), bsz),
                "energy": np.stack(energy_hist).flatten(),
            })
            fig, ax = plt.subplots()
            sns.lineplot(data=plot_data, x="step", y="energy", errorbar="pi", ax=ax)
            plt.show()
            return plot_data

    def generate(
        self,
        n_proposals=1,
        energy_threshold=float("Inf"),
        max_attempts=10000,
        grad_mask=None,
        n_steps=20,
        learning_rate=0.5,
        step_print=10,
        lr_scheduler=True,
        create_plot=False,
    ):
        """
        Run optimization repeatedly until n_proposals sequences below
        energy_threshold are collected or max_attempts is reached.

        Returns:
            dict with keys: states, proposals, energies.
        """
        batch_size, *theta_shape = self.params.theta.shape

        proposals = torch.randn([0, *theta_shape])
        states = torch.randn([0, *theta_shape])
        energies = torch.randn([0])
        attempts = 0

        while proposals.shape[0] < n_proposals and attempts < max_attempts:
            attempts += 1

            self.run(
                n_steps=n_steps,
                learning_rate=learning_rate,
                step_print=step_print,
                lr_scheduler=lr_scheduler,
                grad_mask=grad_mask,
                create_plot=create_plot,
            )

            with torch.no_grad():
                final_states = self.params.theta
                final_samples = self.params.get_sample()
                final_energies = self.energy_fn.energy_calc(final_samples.flatten(0, 1))

                state_bs, energy_bs = final_states.shape[0], final_energies.shape[0]

                if state_bs != energy_bs:
                    rebatch_energies = final_energies.unflatten(0, (energy_bs // state_bs, state_bs))
                    best_sample_idx = rebatch_energies.argmin(dim=0)
                    range_slicer = torch.arange(rebatch_energies.shape[1])
                    final_samples = final_samples[best_sample_idx, range_slicer].squeeze()
                    final_energies = rebatch_energies[best_sample_idx, range_slicer].squeeze()
                else:
                    final_samples = final_samples.squeeze()

                energy_filter = final_energies <= energy_threshold
                final_states = final_states.detach().clone()
                final_samples = final_samples.detach().clone()
                final_energies = final_energies.detach().clone()

            states = torch.cat([states, final_states[energy_filter].squeeze(dim=0).cpu()], dim=0)
            proposals = torch.cat([proposals, final_samples[energy_filter].cpu()], dim=0)
            energies = torch.cat([energies, final_energies[energy_filter].cpu()], dim=0)

            self.params.reset()

        return {
            "states": states[:n_proposals],
            "proposals": proposals[:n_proposals],
            "energies": energies[:n_proposals],
        }


######################
##                  ##
##     Generate     ##
##                  ##
######################


def generate(
    model,
    seq_len,
    device,
    n_samples_train,
    n_steps,
    n_proposals,
    utr_type,
    CELLTYPE_CODES_FORGEN,
    batch_size=1,
    loss_type="square",
    create_plot=False,
):
    """
    Generate optimized sequences for each cell type in CELLTYPE_CODES_FORGEN.

    Args:
        model: Predictor network.
        seq_len: Sequence length.
        device: Torch device.
        n_samples_train: Number of Gumbel samples per step.
        n_steps: Number of optimization steps.
        n_proposals: Number of sequences to generate per cell type.
        utr_type: "utr3" or "utr5".
        CELLTYPE_CODES_FORGEN: Dict mapping cell type name to index.
        batch_size: Number of sequences optimized in parallel.
        loss_type: Energy loss type.

    Returns:
        seq_dict: dict mapping cell type -> list of proposal tensors.
        params: last GumbelSoftmaxParameters instance (for inspection).
    """
    seq_dict = defaultdict(list)

    for target_feature in tqdm(CELLTYPE_CODES_FORGEN):
        test_seq = gen_random_seqs(seq_len, utr_type, target_feature, batch_size, device)
        params = GumbelSoftmaxParameters(
            data=test_seq,
            n_samples=n_samples_train,
            tau=0.9,
            use_affine=False,
        ).to(device)

        energy = MinGapEnergy(
            model,
            device=device,
            seq_len=seq_len,
            target_feature=CELLTYPE_CODES_FORGEN[target_feature],
            a_min=-math.inf,
            a_max=math.inf,
            target_alpha=1,
            utr_type=utr_type,
            loss_type=loss_type,
        )

        fsp = FastSeqProp(energy, params)
        proposals = fsp.generate(
            n_proposals=n_proposals,
            n_steps=n_steps,
            lr_scheduler=True,
            create_plot=create_plot,
        )
        out_seqs = proposals["proposals"].squeeze(dim=1)
        seq_dict[target_feature].append(out_seqs)

    return seq_dict, params
