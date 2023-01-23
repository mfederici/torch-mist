from torchvision.models.resnet import BasicBlock
from torch import nn
import torch


class OmniglotResNet(nn.Module):
    def __init__(
            self,
            n_images: int = 1,
            out_dim: int = 128,
    ):
        super().__init__()

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(in_channels=n_images, out_channels=32 * n_images, kernel_size=5, stride=3),
            BasicBlock(
                32 * n_images, 32 * n_images, stride=2,
                norm_layer=lambda x: nn.LayerNorm([x, 4, 4], eps=1e-6),
                downsample=nn.Sequential(
                    nn.Conv2d(in_channels=32 * n_images, out_channels=32 * n_images, kernel_size=1, stride=(2, 2),
                              bias=False),
                    nn.LayerNorm([32 * n_images, 4, 4], eps=1e-6),
                )
            ),
            BasicBlock(
                32 * n_images, 32 * n_images,
                norm_layer=lambda x: nn.LayerNorm([x, 4, 4], eps=1e-6),
            ),
            BasicBlock(
                32 * n_images, 64 * n_images, stride=2,
                norm_layer=lambda x: nn.LayerNorm([x, 2, 2], eps=1e-6),
                downsample=nn.Sequential(
                    nn.Conv2d(in_channels=32 * n_images, out_channels=64 * n_images, kernel_size=1, stride=2,
                              bias=False),
                    nn.LayerNorm([64 * n_images, 2, 2], eps=1e-6),
                )
            ),
            BasicBlock(
                64 * n_images, 64 * n_images,
                norm_layer=lambda x: nn.LayerNorm([x, 2, 2], eps=1e-6),
            ),
            BasicBlock(
                    64 * n_images, 64 * n_images,
                    norm_layer=lambda x: nn.LayerNorm([x, 2, 2], eps=1e-6),
                )
        )

        self.proj_head = nn.Linear(64 * n_images, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 4:
            batch_shapes = x.shape[:-3]
            x = x.flatten(start_dim=0, end_dim=-4)
        else:
            batch_shapes = None
        x = self.conv_encoder(x)
        x = x.sum(dim=[2, 3])
        x = self.proj_head(x)

        if batch_shapes is not None:
            x = x.view(*batch_shapes, *x.shape[1:])
        return x
