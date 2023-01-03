from functools import partial
from typing import Optional, Callable, List

import torch
import torchvision.models as tv_models
from torch import nn
from torchvision.models.vision_transformer import ConvStemConfig


class VisionTransformer(tv_models.VisionTransformer):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_classes: int = 1000,
            representation_size: Optional[int] = None,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
            conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=num_classes,
            representation_size=representation_size,
            norm_layer=norm_layer,
            conv_stem_configs=conv_stem_configs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 4:
            batch_shapes = x.shape[:-3]
            x = x.flatten(start_dim=0, end_dim=-4)
        else:
            batch_shapes = None
        x = super().forward(x)

        if batch_shapes is not None:
            x = x.view(*batch_shapes, *x.shape[1:])
        return x


