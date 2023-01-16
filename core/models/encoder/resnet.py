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

        in_channels = 1

        sizes = [
            0,
            [4, 2],
            [9, 5, 3],
            [14, 7, 4]
        ]

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32 * n_images, kernel_size=5, stride=3),
            BasicBlock(
                32 * n_images, 32 * n_images, stride=2,
                norm_layer=lambda x: nn.LayerNorm([x, sizes[n_images][0], sizes[n_images][0]], eps=1e-6),
                downsample=nn.Sequential(
                    nn.Conv2d(in_channels=32 * n_images, out_channels=32 * n_images, kernel_size=1, stride=(2, 2),
                              bias=False),
                    nn.LayerNorm([32 * n_images, sizes[n_images][0], sizes[n_images][0]], eps=1e-6),
                )
            ),
            BasicBlock(
                32 * n_images, 32 * n_images,
                norm_layer=lambda x: nn.LayerNorm([x, sizes[n_images][0], sizes[n_images][0]], eps=1e-6),
            ),
            BasicBlock(
                32 * n_images, 64 * n_images, stride=2,
                norm_layer=lambda x: nn.LayerNorm([x, sizes[n_images][1], sizes[n_images][1]], eps=1e-6),
                downsample=nn.Sequential(
                    nn.Conv2d(in_channels=32 * n_images, out_channels=64 * n_images, kernel_size=1, stride=2,
                              bias=False),
                    nn.LayerNorm([64 * n_images, sizes[n_images][1], sizes[n_images][1]], eps=1e-6),
                )
            ),
            BasicBlock(
                64 * n_images, 64 * n_images,
                norm_layer=lambda x: nn.LayerNorm([x, sizes[n_images][1], sizes[n_images][1]], eps=1e-6),
            ),
        )

        if n_images == 1:
            self.conv_encoder.add_module(
                "5",
                BasicBlock(
                    64 * n_images, 64 * n_images,
                    norm_layer=lambda x: nn.LayerNorm([x, sizes[n_images][1], sizes[n_images][1]], eps=1e-6),
                )
            )
            self.conv_encoder.add_module(
                "6",
                BasicBlock(
                    64 * n_images, 64 * n_images,
                    norm_layer=lambda x: nn.LayerNorm([x, sizes[n_images][1], sizes[n_images][1]], eps=1e-6),
                )
            )
        else:
            self.conv_encoder.add_module(
                "5",
                BasicBlock(
                    64 * n_images, 64 * n_images, stride=2,
                    norm_layer=lambda x: nn.LayerNorm([x, sizes[n_images][2], sizes[n_images][2]], eps=1e-6),
                    downsample=nn.Sequential(
                        nn.Conv2d(in_channels=64 * n_images, out_channels=64 * n_images, kernel_size=1, stride=2,
                                  bias=False),
                        nn.LayerNorm([64 * n_images, sizes[n_images][2], sizes[n_images][2]], eps=1e-6),
                    )
                )
            )
            self.conv_encoder.add_module(
                "6",
                BasicBlock(
                    64 * n_images, 64 * n_images,
                    norm_layer=lambda x: nn.LayerNorm([x, sizes[n_images][2], sizes[n_images][2]], eps=1e-6),
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

