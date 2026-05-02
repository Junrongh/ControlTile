from typing import List, Tuple

import torch
from torch import nn
from diffusers import FluxTransformer2DModel

class XEmbedder(nn.Module):
    def __init__(self, in_channels: List[Tuple[str, int]], original_x_embedder: nn.Linear):
        super().__init__()

        init_input_channels = original_x_embedder.in_features
        output_channels = original_x_embedder.out_features
        new_input_channels = init_input_channels
        use_bias = original_x_embedder.bias is not None
        for feature in in_channels:
            _, channels, weight = feature
            new_input_channels += channels

        with torch.no_grad():
            new_linear = nn.Linear(
                new_input_channels,
                output_channels,
                bias=use_bias,
            )
            new_linear.weight.zero_()
            new_linear.weight[:, :init_input_channels].copy_(original_x_embedder.weight)
            if use_bias:
                new_linear.bias.copy_(original_x_embedder.bias)
            self.x_embedder = new_linear

        self.in_features = new_input_channels
        self.out_features = output_channels
        self.use_bias = use_bias
        self.init_input_channels = init_input_channels
        
    def forward(self, x):
        out = self.x_embedder(x)
        return out

def set_x_embedder(transformer: FluxTransformer2DModel, additional_embedder_channels: List[Tuple[str, int]]):
    transformer.x_embedder = XEmbedder(additional_embedder_channels, transformer.x_embedder)
