import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Union
from src.config.config_loader import (
    ConvLayerConfig,
    PoolLayerConfig,
    ResBlockConfig,
    DenseLayerConfig,
)


def get_activation(activation_name: Optional[str]) -> nn.Module:
    """
    Returns the activation module corresponding to the given name.
    If activation_name is None or "none", returns an identity.
    """
    match activation_name.lower():
        case None | "none":
            return nn.Identity()
        case "relu":
            return nn.ReLU(inplace=True)
        case "tanh":
            return nn.Tanh()
        case "sigmoid":
            return nn.Sigmoid()
        case _:
            raise ValueError(f"Unknown activation function: {activation_name}")


class ConvLayer(nn.Module):
    """
    A simple convolution layer with optional activation.
    This corresponds to a `ConvLayerConfig`.
    """

    def __init__(self, config: ConvLayerConfig, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
        )
        self.activation = get_activation(config.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class PoolLayer(nn.Module):
    """
    A pooling layer, which can be either max or average pooling,
    as defined by `pool_type` in the config.
    """

    def __init__(self, config: PoolLayerConfig):
        super().__init__()
        match config.pool_type.lower():
            case "max":
                self.pool = nn.MaxPool2d(
                    kernel_size=config.kernel_size, stride=config.stride
                )
            case "avg":
                self.pool = nn.AvgPool2d(
                    kernel_size=config.kernel_size, stride=config.stride
                )
            case _:
                raise ValueError(f"Unknown pool_type: {config.pool_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class ResBlock(nn.Module):
    """
    A residual block:
      - 2 convolutions
      - A skip connection
      - Optional downsample if needed
      - Optional pooling (based on config.pool_kernel_size, pool_stride)
    """

    def __init__(self, config: ResBlockConfig, in_channels: int):
        super().__init__()

        self.activation = get_activation(config.activation)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(config.out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=config.out_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(config.out_channels)

        # If the input and output channels differ or stride > 1,
        # we need a conv to match the dimensions in the skip.
        self.downsample = None
        if (in_channels != config.out_channels) or (config.stride != 1):
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=config.out_channels,
                    kernel_size=1,
                    stride=config.stride,
                    bias=False,
                ),
                nn.BatchNorm2d(config.out_channels),
            )

        # Optional pooling as specified by the config
        if config.pool_kernel_size > 1:
            self.pool = nn.MaxPool2d(
                kernel_size=config.pool_kernel_size,
                stride=config.pool_stride,
            )
        else:
            self.pool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        # Second conv
        out = self.conv2(out)
        out = self.bn2(out)

        # Downsample skip if needed
        if self.downsample is not None:
            residual = self.downsample(residual)

        # Add skip
        out += residual
        out = self.activation(out)

        # Pooling if configured
        out = self.pool(out)
        return out


def build_downsample_layer(
    layer_config: Union[ConvLayerConfig, PoolLayerConfig, ResBlockConfig],
    in_channels: int,
) -> Tuple[nn.Module, int]:
    """
    Given a layer config (conv, pool, or res_block) and the current in_channels,
    return:
      (created_layer, output_channels)
    so we can keep track of how the channel dimension changes.
    """
    if layer_config.type == "conv_layer":
        conv_layer = ConvLayer(layer_config, in_channels)
        out_channels = layer_config.out_channels
        return conv_layer, out_channels

    elif layer_config.type == "pool_layer":
        # Pooling does not change the channel dimension
        pool_layer = PoolLayer(layer_config)
        return pool_layer, in_channels

    elif layer_config.type == "res_block":
        res_block = ResBlock(layer_config, in_channels)
        out_channels = layer_config.out_channels
        return res_block, out_channels

    else:
        raise ValueError(f"Unknown layer type: {layer_config.type}")
