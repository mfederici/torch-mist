from typing import Optional, Any, Dict

import torch
from pyro.distributions import ConditionalDistribution
from torch import nn

from torch_mist.quantization import QuantizationFunction
from torch_mist.quantization.functions import (
    LearnableVectorQuantization,
)

VERSIONS = ["v1", "v2"]


class VQVAE(LearnableVectorQuantization):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: ConditionalDistribution,
        initial_vectors: Optional[torch.Tensor] = None,
        n_bins: Optional[int] = None,
        quantization_dim: Optional[int] = None,
        beta: float = 0.25,
        gamma: float = 0.99,
        version: str = "v2",
        **train_params,
    ):
        if "minimize" not in train_params:
            train_params["minimize"] = True

        if not (version in VERSIONS):
            raise ValueError(f"Please choose version among {VERSIONS}.")
        super().__init__(
            vectors=initial_vectors,
            n_bins=n_bins,
            quantization_dim=quantization_dim,
            **train_params,
        )
        self.encoder = encoder
        self.decoder = decoder

        self.beta = beta
        self.gamma = gamma
        self.version = version

        initial_assignments = torch.ones(self.n_bins) / self.n_bins
        self.register_buffer("assignments", initial_assignments)
        self.min_assignment = 0.1 / self.n_bins

    def quantize(self, x: torch.Tensor) -> torch.LongTensor:
        z = self.encoder(x)
        assert z.shape[-1] == self.vectors.shape[-1]
        return super().quantize(z)

    def _update_vectors(self, indices: torch.LongTensor, z: torch.Tensor):
        # Compute the one-hot representation of the indices [N*M, n_bins]
        flat_indices = indices.view(-1)
        one_hot = torch.nn.functional.one_hot(
            flat_indices, self.n_bins
        ).float()

        # Compute the number of datapoints assigned to each bin [n_bins]
        assignments = one_hot.mean(0)

        self.assignments = (
            self.gamma * self.assignments + (1 - self.gamma) * assignments
        )

        # Compute the mean vector for each assignment [n_bins, z_dim]
        N = flat_indices.shape[0]
        mean_vector = (one_hot.unsqueeze(-1) * z.unsqueeze(-2)).sum(0)
        mean_vector = mean_vector / (assignments.unsqueeze(-1) * N)

        not_assigned = [
            i.item()
            for i in torch.arange(self.n_bins)[
                self.assignments < self.min_assignment
            ].long()
        ]

        while len(not_assigned) > 0:
            not_assigned_idx = not_assigned.pop()
            # Select the vector with the highest assignment
            max_idx = torch.argmax(self.assignments)
            # set the centroid to the vector with the highest assignment + some noise
            noise = torch.randn_like(self.vectors[max_idx]) * 1e-5
            self.vectors[not_assigned_idx] = self.vectors[max_idx] + noise
            # set the new assignements to half the max assignment
            new_assignment = self.assignments[max_idx] / 2
            self.assignments[not_assigned_idx] = new_assignment
            self.assignments[max_idx] = new_assignment

        mean_vector[assignments == 0] = self.vectors[assignments == 0]

        self.vectors = (
            self.gamma * self.vectors + (1 - self.gamma) * mean_vector
        )

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        indices = super().quantize(z)

        vectors = self.vectors[indices]
        z_q = z + (vectors - z).detach()

        assert (
            z_q.shape == z.shape == vectors.shape
        ), f"{z_q.shape} != {z.shape} != {vectors.shape}"

        reconstruction_loss = -self.decoder.condition(z_q).log_prob(x)
        assert (
            reconstruction_loss.shape == x.shape[:-1]
        ), f"{reconstruction_loss.shape} != {x.shape[:-1]}"

        reconstruction_loss = reconstruction_loss.mean()
        commitment_loss = torch.mean((vectors.detach() - z) ** 2)

        loss = reconstruction_loss + self.beta * commitment_loss

        if self.training and not self.trained:
            if self.version == "v1":
                codebook_loss = torch.mean((vectors - z.detach()) ** 2)
                loss += codebook_loss
            elif self.version == "v2":
                self._update_vectors(indices, z)
            else:
                raise NotImplementedError()

        return loss
