

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from copy import deepcopy
from tqdm.auto import tqdm

from fm_dataset import idx2cell_type, cell_type_mapper, OneHotSeqEncoder


##############################
####### NEURAL NETWORK #######
##############################

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.register_buffer('W', torch.randn(embed_dim // 2) * scale)
        self.register_parameter('uncond_emb', nn.Parameter(torch.randn(embed_dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten()
        unconditional_mask = x.isinf()

        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        output = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        output[unconditional_mask, :] = self.uncond_emb
        return output


class CNNModel(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_dim: int,
                 num_cnn_stacks: int,
                 dropout: float,
                 num_cell_types: int,
                 time_fourier_scale: float = 30,
                 expression_fourier_scale: float = 30,
                 ) -> None:

        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_cnn_stacks = num_cnn_stacks
        self.dropout = dropout
        self.num_cell_types = num_cell_types

        inp_size = self.input_size * 2

        self.linear = nn.Conv1d(inp_size, self.hidden_dim, kernel_size=9, padding=4)
        self.num_layers = 5 * self.num_cnn_stacks

        basic_block = [
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, padding=4),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=4, padding=16),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=16, padding=64),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=9, dilation=64, padding=256)
        ]

        self.convs = nn.ModuleList([deepcopy(layer) for layer in basic_block
                                    for _ in range(self.num_cnn_stacks)])

        self.time_embedder = nn.Sequential(
            GaussianFourierProjection(
                embed_dim=self.hidden_dim,
                scale=time_fourier_scale
            ),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.time_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim)
             for _ in range(self.num_layers)]
        )

        self.cell_type_embedder = nn.Embedding(
            num_embeddings=self.num_cell_types + 1,
            embedding_dim=self.hidden_dim
        )

        self.cell_type_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim)
             for _ in range(self.num_layers)]
        )

        self.expr_embedder = nn.Sequential(
            GaussianFourierProjection(
                embed_dim=self.hidden_dim,
                scale=expression_fourier_scale
            ),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.expr_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim)
             for _ in range(self.num_layers)]
        )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(self.hidden_dim, self.input_size, kernel_size=1)
        )

        self.dropout = nn.Dropout(self.dropout)

    def forward(self,
                seq: torch.Tensor,
                time: torch.Tensor,
                cell_type: torch.Tensor,
                expression: torch.Tensor
                ) -> torch.Tensor:

        feat = seq.permute(0, 2, 1)
        feat = F.relu(self.linear(feat))

        time_emb = F.relu(self.time_embedder(time))
        cell_type_emb = self.cell_type_embedder(cell_type)
        expr_emb = self.expr_embedder(expression)

        for i in range(self.num_layers):
            # Apply dropout to the current feature map
            h = self.dropout(feat.clone())

            # Add time embedding
            h = h + self.time_layers[i](time_emb)[:, :, None]

            # Add desired class embedding
            h = h + self.cell_type_layers[i](cell_type_emb)[:, :, None]

            # Add desired expression embedding
            h = h + self.expr_layers[i](expr_emb)[:, :, None]

            # Apply normalization
            h = self.norms[i](h.permute(0, 2, 1))

            # Apply convolution
            h = F.relu(self.convs[i](h.permute(0, 2, 1)))

            # Skip connection if possible
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h

        # Transform output into correct size
        feat = self.final_conv(feat)
        feat = feat.transpose(1, 2)
        return feat


##############################
####### FLOW MATCHING ########
##############################


class BetaTimeSampler:
    def __init__(self, alpha: float, beta: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.beta_distr = torch.distributions.Beta(
            concentration1=torch.Tensor([self.alpha]),
            concentration0=torch.Tensor([self.beta])
        )

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.beta_distr.sample((num_samples,)).view(-1)

    def plot_hist(self,
                  num_samples: int = 10000,
                  bins: int = 50) -> None:
        plt.hist(self.sample(num_samples), bins=bins)


class FlowMatcher(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 seq_length: int,
                 prior_pseudocount: float,
                 flow_temp: float,
                 cls_free_noclass_ratio: float,
                 time_sampler: object,
                 device: str | torch.device
                 ) -> None:
        super().__init__()

        self.model = model
        self.seq_length = seq_length
        self.prior_pseudocount = prior_pseudocount
        self.flow_temp = flow_temp
        self.cls_free_noclass_ratio = cls_free_noclass_ratio
        self.time_sampler = time_sampler
        self.device = device
        self.model.to(self.device)

        self.nan_inf_counter = 0
        self.inf_counter = 0

        self.crossent_loss = torch.nn.CrossEntropyLoss(reduction='none')

        # TODO: CHANGE TIME SAMPLER!!!

    @staticmethod
    def expand_simplex(xt, time):
        time = time[:, None, None]
        return torch.cat([xt * (1 - time), xt * time], -1)  # , prior_weights

    def sample_cond_prob_path(self, seq: torch.Tensor
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, L = seq.shape
        assert C == 4

        # t = torch.rand(B, device=seq.device)
        t = self.time_sampler.sample(num_samples=B).to(self.device)
        dirichlet = torch.distributions.Dirichlet(torch.ones(C, device=seq.device))
        x0 = dirichlet.sample((B, L))
        x1 = seq.transpose(1, 2)
        # print(x0.shape, x1.shape)
        xt = t[:, None, None] * x1 + (1 - t[:, None, None]) * x0
        alphas = t
        return xt, alphas

    def training_step(
            self,
            batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:

        batch = self.transfer_batch(batch)

        xt, time = self.sample_cond_prob_path(batch['ohe_seq'])
        xt_inp = self.expand_simplex(xt, time)

        non_class_mask = torch.ones(
            batch['ohe_seq'].shape[0]
        ).uniform_(0, 1) < self.cls_free_noclass_ratio

        batch['cell_type'][non_class_mask] = self.model.num_cell_types
        batch['expression'][non_class_mask] = torch.inf


        logits = self.model(
            xt_inp,
            time=time,
            cell_type=batch['cell_type'].squeeze(-1).long(),
            expression=batch['expression']
        ).transpose(1, 2)

        # print(logits.shape)

        losses = torch.nn.functional.cross_entropy(
            logits, batch['ohe_seq'].argmax(dim=1), reduction='none')
        # losses = losses.mean(-1)
        return losses.mean()

    @torch.no_grad()
    def sample(
            self,
            num_samples: int,
            num_integration_steps: int,
            cell_type: torch.Tensor | int,
            expression: torch.Tensor | float | int,
            guidance_scale: float,
            verbose: bool = False,
            show_pbar: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        B, L = num_samples, self.seq_length

        if isinstance(cell_type, int):
            cell_type = torch.tensor([cell_type]).expand(B)

        if isinstance(expression, float) or isinstance(expression, int):
            expression = torch.Tensor([expression]).expand(B)

        cell_type = cell_type.to(self.device)
        expression = expression.to(self.device)

        K = self.model.input_size
        x0 = torch.distributions.Dirichlet(
            torch.ones(B, L, self.model.input_size, device=self.device)).sample()
        eye = torch.eye(K).to(x0)
        xt = x0.clone()

        # t_span = torch.linspace(0, 1, self.num_integration_steps, device=self.device)
        t_span = torch.linspace(
            0, 1,
            num_integration_steps,
            device=self.device
        )

        progress_bar = tqdm(
            zip(t_span[:-1], t_span[1:]),
            total=num_integration_steps - 1
        ) if show_pbar else zip(t_span[:-1], t_span[1:])

        for s, t in progress_bar:
            xt_expanded = self.expand_simplex(
                xt, s[None].expand(B))

            ## Conditional probabilities

            logits_cond = self.model(
                xt_expanded,
                time=s[None].expand(B),
                cell_type=cell_type,
                expression=expression,
            )
            probs_cond = torch.nn.functional.softmax(logits_cond / self.flow_temp, -1)

            ## Unconditional probabilities

            logits_uncond = self.model(
                xt_expanded,
                time=s[None].expand(B),
                cell_type=(torch.ones(B, device=self.device) * self.model.num_cell_types).long(),
                expression=torch.Tensor([torch.inf]).expand(B).to(self.device),
            )
            probs_uncond = torch.nn.functional.softmax(logits_uncond / self.flow_temp, -1)

            ## Next step calculation

            flows = (eye - xt.unsqueeze(-1)) / (1 - s)

            redistribution_cond = (probs_cond.unsqueeze(-2) * flows).sum(-1)
            redistribution_uncond = (probs_uncond.unsqueeze(-2) * flows).sum(-1)

            # print(redistribution_cond.sum(dim=-1).shape)
            # print(redistribution_cond.sum(dim=-1)[0])

            # print(redistribution_cond.sum(dim=-1).abs().max())
            # print(redistribution_uncond.sum(dim=-1).abs().max())

            assert torch.allclose(
                redistribution_cond.sum(dim=-1),
                torch.zeros_like(redistribution_cond.sum(dim=-1)).to(self.device),
                atol=1e-3
            )

            assert torch.allclose(
                redistribution_uncond.sum(dim=-1),
                torch.zeros_like(redistribution_uncond.sum(dim=-1)).to(self.device),
                atol=1e-3
            )

            final_redistr = guidance_scale * redistribution_cond + (1 - guidance_scale) * redistribution_uncond
            xt = xt + final_redistr * (t - s)

        if verbose:
            return xt.transpose(1, 2), x0.transpose(1, 2)

        return xt.transpose(1, 2)

    def transfer_batch(
            self,
            batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        for key, tns in batch.items():
            batch[key] = tns.to(self.device, dtype=torch.float)
        return batch

    def run_training(
            self,
            dataloader: DataLoader,
            max_lr: float,
            num_epochs: int
    ) -> list[float]:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=max_lr,
            weight_decay=1e-5
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            epochs=num_epochs,
            steps_per_epoch=len(dataloader),
        )

        loss_history = []

        for epoch in tqdm(range(num_epochs)):
            epoch_progress_bar = tqdm(dataloader)
            for batch in epoch_progress_bar:
                optimizer.zero_grad()
                loss = self.training_step(batch)
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_progress_bar.set_postfix(loss=loss.item())
                loss_history.append(loss.item())

        return loss_history


class UTRGenerator(nn.Module):
    def __init__(self,
                 utr_type: int,
                 alphabet_size: int = 4,
                 device: torch.device | str = 'cpu',
                 #  model_kws: dict[str, tp.Any] | None = None,
                 ) -> None:
        super().__init__()

        assert utr_type in [5, 3], "UTR type must be 5 or 3"
        self.utr_type = utr_type

        self.alphabet_size = alphabet_size

        self.num_cell_types = 5 if self.utr_type == 5 else 7

        self.device = device
        super().to(self.device)
        self.eval()

        self.ohse = OneHotSeqEncoder()

    def score_generator(
        self,
        generator: nn.Module,
        generation_method_name: str,
        total_samples: int,
        batch_size: int,
        **generation_kwargs
    ):

        tables = self._latest_tables = []
        num_cell_types = 5 if self.utr_type == 5 else 6
        for cell_type_idx in tqdm(range(num_cell_types)):
            generation_kwargs['cell_type'] = cell_type_idx
            table = self.score_generator_one_cell_type(
                generator,
                generation_method_name,
                total_samples,
                batch_size,
                **generation_kwargs
            )
            tables.append(table)

        sequence_table = pd.concat(tables, axis=0)
        return None, sequence_table

    def score_generator_one_cell_type(
        self,
        generator: nn.Module,
        generation_method_name: str,
        total_samples: int,
        batch_size: int,
        **generation_kwargs
    ):

        generation_kwargs = deepcopy(generation_kwargs)
        expressions = torch.zeros(total_samples,)

        if self.utr_type == 3:
            expressions = expressions.uniform_(1.8, 3.2)
        else:
            expressions = expressions.uniform_(2, 3)

        expr_loader = DataLoader(expressions, batch_size=batch_size)

        all_desired_scores = []
        all_generated_sequences = []

        for batch in tqdm(expr_loader, total=len(expr_loader)):
            generation_kwargs['expression'] = batch.to(generator.device)
            generation_kwargs['num_samples'] = len(batch)
            generated = getattr(generator, generation_method_name)(
                **generation_kwargs).detach().cpu()

            # We want to score sequences, not probability
            #  distributions. So we apply argmax and then
            #  one-hot-encode the result.
            generated = self.concentrate_probability_mass(
                generated, self.alphabet_size).float()

            desired_scores = batch.detach().cpu()

            all_desired_scores.extend(list(desired_scores))

            decoded_seqs = self.ohse.decode(generated.argmax(dim=1))
            all_generated_sequences.extend(decoded_seqs)

        all_desired_scores = np.array(all_desired_scores).reshape(
            len(all_desired_scores), )

        cell_type_name = cell_type_mapper[
            idx2cell_type[
                generation_kwargs['cell_type']
            ]
        ]

        table = pd.DataFrame().from_dict(
            {
                'seq': all_generated_sequences,
                'desired_expr': all_desired_scores,
                'cell_type': [cell_type_name] * len(all_generated_sequences)
            }
        )
        return table

    @staticmethod
    def concentrate_probability_mass(
            sequence_batch: torch.Tensor,
            alphabet_size: int
    ) -> torch.Tensor:

        token_indices = sequence_batch.argmax(dim=1)
        degenerate_simplices = F.one_hot(
            token_indices, alphabet_size).transpose(1, 2)
        return degenerate_simplices
