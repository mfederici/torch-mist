from torch.distributions import Distribution


class ShuffledBatchDistribution(Distribution):
    def __init__(self, y: torch.Tensor):
        super().__init__(validate_args=False)
        assert y.shape[1] == 1, f"The input tensor must have shape [N, 1, ...]. {y.shape} was given."
        self.y = y

    def rsample(self, sample_shape=torch.Size()):
        assert sample_shape.ndim == 1, "Only 1D sample shapes are supported."
        n_samples = sample_shape[0]

        # if the sample shape is zero we use the whole batch as negatives
        if n_samples == 0:
            # simply unsqueeze an empty dimension at the beginning (we don't repeat to save memory)
            return self.y[:, 0].unsqueeze(0)
        elif n_samples < 0:
            n_samples = self.y.shape[0] + n_samples

        assert n_samples > 0, "Invalid sample shape."

        N = self.y.shape[0]
        # This indexing operation takes care of selecting the appropriate (off-diagonal) y
        idx = torch.arange(N * n_samples).to(self.y.device).view(N, n_samples).long()
        idx = (idx % n_samples + torch.div(idx, n_samples, rounding_mode='floor') + 1) % N
        return self.y[:, 0][idx]

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape).detach()







