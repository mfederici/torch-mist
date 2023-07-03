from typing import Optional, List, Iterator

import torch
from pyro.distributions import ConditionalDistribution
from torch import nn

from torch_mist.quantization.functions import LearnableVectorQuantization, QuantizationFunction

INITIAL_PATIENCE = 10.0


class VQVAE(nn.Module):
    def __init__(
            self,
            encoder: LearnableVectorQuantization,
            decoder: ConditionalDistribution,
            beta: float = 0.25,
            gamma: float = 0.99,
            cross_modal: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        n_bins = encoder.n_bins

        self.beta = beta
        self.gamma = gamma
        self.cross_modal = cross_modal

        initial_assignments = torch.zeros(n_bins) + INITIAL_PATIENCE
        self.register_buffer('assignments', initial_assignments)
        self.min_assignment = 0.1/n_bins

    @property
    def quantization(self) -> QuantizationFunction:
        return self.encoder

    def loss(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert not self.cross_modal or y is not None, "y must be provided if cross_modal is True"

        z = self.encoder.net(x)
        indices = self.encoder.codebook_lookup(z)

        quantized = self.encoder.vectors[indices]
        z_q = z + (quantized - z).detach()

        assert z_q.shape == z.shape == quantized.shape, f"{z_q.shape} != {z.shape} != {quantized.shape}"

        if self.cross_modal:
            reconstruction_loss = -self.decoder.condition(z_q).log_prob(y)
            assert reconstruction_loss.shape == y.shape[:-1], f"{reconstruction_loss.shape} != {y.shape[:-1]}"
            reconstruction_loss = reconstruction_loss.mean()
        else:
            reconstruction_loss = -self.decoder.condition(z_q).log_prob(x)
            assert reconstruction_loss.shape == x.shape[:-1], f"{reconstruction_loss.shape} != {x.shape[:-1]}"
            reconstruction_loss = reconstruction_loss.mean()

        if self.training:
            # Compute the one-hot representation of the indices [N*M, n_bins]
            flat_indices = indices.view(-1)
            one_hot = torch.nn.functional.one_hot(flat_indices, self.encoder.n_bins).float()

            # Compute the number of datapoints assigned to each bin [n_bins]
            assignments = one_hot.mean(0)

            if self.training:
                self.assignments = self.gamma * self.assignments + (1 - self.gamma) * assignments
            # assignments = self.assignments

            # Compute the mean vector for each assignment [n_bins, z_dim]
            N = flat_indices.shape[0]
            mean_vector = (one_hot.unsqueeze(-1) * z.unsqueeze(-2)).sum(0)
            mean_vector = mean_vector / (assignments.unsqueeze(-1) * N)

            not_assigned = [i.item() for i in torch.arange(self.encoder.n_bins)[self.assignments * N < self.min_assignment].long()]
            while len(not_assigned) > 0:
                not_assigned_idx = not_assigned.pop()
                # Select the vector with the highest assignment
                max_idx = torch.argmax(assignments)
                # set the centroid to the vector with the highest assignment + some noise
                noise = torch.randn_like(self.encoder.vectors[max_idx]) * 1e-5
                self.encoder.vectors[not_assigned_idx] = self.encoder.vectors[max_idx] + noise
                # set the new assignements to half the max assignment
                new_assignment = self.assignments[max_idx] / 2
                self.assignments[not_assigned_idx] = new_assignment
                self.assignments[max_idx] = new_assignment

            mean_vector[assignments == 0] = self.encoder.vectors[assignments == 0]
            # print((self.assignments * N < self.min_assignment).sum().item())

            self.encoder.vectors = self.gamma * self.encoder.vectors + (1 - self.gamma) * mean_vector

        commitment_loss = torch.mean((quantized.detach() - z)**2)
        # codebook_loss = torch.mean((quantized - z.detach())**2)
        loss = reconstruction_loss + self.beta * commitment_loss #+ codebook_loss
        return loss


def vqvae(
        x_dim: int,
        quantization_dim: int,
        n_bins: int,
        hidden_dims: List[int],
        y_dim: Optional[int] = None,
        cross_modal: bool = False,
        decoder_transform_params: Optional[dict] = None,
        beta: float = 0.2,
) -> VQVAE:
    assert not cross_modal or y_dim is not None, "If cross_modal is True, y_dim must be provided"

    from torch_mist.quantization import learnable_vector_quantization
    from torch_mist.distributions.utils import conditional_transformed_normal

    quantization = learnable_vector_quantization(
        input_dim=x_dim,
        quantization_dim=quantization_dim,
        hidden_dims=hidden_dims,
        n_bins=n_bins
    )

    decoder = conditional_transformed_normal(
            input_dim=x_dim if not cross_modal else y_dim,
            context_dim=quantization_dim,
            transform_name='conditional_linear',
            transform_params=decoder_transform_params
    )

    return VQVAE(
        encoder=quantization,
        decoder=decoder,
        cross_modal=cross_modal,
        beta=beta
    )
