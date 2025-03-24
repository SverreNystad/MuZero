from enum import StrEnum
import os
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field
import yaml

from src.environments.car_racing import CarRacingConfig
from src.environments.connect_four import ConnectFourConfig


CONFIG_PATH = os.path.dirname(__file__)


EnvironmentConfig = Union[CarRacingConfig, ConnectFourConfig]


class SelectionStrategyType(StrEnum):
    uct = "uct"
    puct = "puct"


class MCTSConfig(BaseModel):
    selection_strategy: SelectionStrategyType = SelectionStrategyType.puct
    max_iterations: int
    max_time: int
    depth: int = 5
    discount_factor: float = 1.0


class ConvLayerConfig(BaseModel):
    type: Literal["conv_layer"] = "conv_layer"
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    activation: Optional[str] = "relu"  # e.g. "relu", "tanh", "sigmoid", "none"


class PoolLayerConfig(BaseModel):
    type: Literal["pool_layer"] = "pool_layer"
    kernel_size: int
    stride: int
    pool_type: str = "max"  # e.g. "max", "avg"


class ResBlockConfig(BaseModel):
    type: Literal["res_block"] = "res_block"
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    activation: str
    pool_kernel_size: int
    pool_stride: int


class DenseLayerConfig(BaseModel):
    out_features: Optional[int] = None  # for conv or linear layers
    activation: Optional[str] = "relu"  # e.g. "relu", "tanh", "sigmoid", "none"


DownsampleLayerConfig = Annotated[
    Union[ConvLayerConfig, PoolLayerConfig, ResBlockConfig], Field(discriminator="type")
]


class RepresentationNetworkConfig(BaseModel):
    downsample: list[DownsampleLayerConfig]
    res_net: list[ResBlockConfig]


class DynamicsNetworkConfig(BaseModel):
    res_net: list[ResBlockConfig]
    reward_net: list[DenseLayerConfig]


class PredictionNetworkConfig(BaseModel):
    res_net: list[ResBlockConfig]
    value_net: list[DenseLayerConfig]
    policy_net: list[DenseLayerConfig]


class NetworksConfig(BaseModel):
    latent_shape: tuple[int, int, int]
    representation: RepresentationNetworkConfig
    dynamics: DynamicsNetworkConfig
    prediction: PredictionNetworkConfig


class TrainingDataGeneratorConfig(BaseModel):
    num_episodes: int
    max_steps_per_episode: int
    total_time: int
    mcts: MCTSConfig


class TrainingConfig(BaseModel):
    learning_rate: float
    batch_size: int
    epochs: int
    betas: tuple[float, float]
    roll_ahead: int
    look_back: int
    mini_batch_size: int


class Configuration(BaseModel):
    environment: EnvironmentConfig = Field(..., discriminator="type")
    networks: NetworksConfig
    training_data_generator: TrainingDataGeneratorConfig
    training: TrainingConfig


def load_config(filename: str) -> Configuration:
    """
    Load a configuration file from the config directory.
    """
    path = os.path.join(CONFIG_PATH, filename)
    with open(path, "r") as file:
        raw_config = yaml.safe_load(file)
    return Configuration(**raw_config)
