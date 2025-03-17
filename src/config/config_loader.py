import os
from typing import Optional, Union

from pydantic import BaseModel, Field
import yaml

from src.environments.car_racing import CarRacingConfig
from src.environments.connect_four import ConnectFourConfig
from src.search.factory import MCTSConfig


CONFIG_PATH = os.path.dirname(__file__)


EnvironmentConfig = Union[CarRacingConfig, ConnectFourConfig]


class LayerConfig(BaseModel):
    type: str  # e.g. "conv", "linear"
    out_channels: Optional[int] = None  # for conv or linear layers
    kernel_size: Optional[int] = None  # for conv layers
    stride: Optional[int] = 1  # for conv layers
    padding: Optional[int] = 0  # for conv layers
    activation: Optional[str] = "relu"  # e.g. "relu", "tanh", "sigmoid", "none"


class RepresentationNetworkConfig(BaseModel):
    latent_dim: int
    layers: list[LayerConfig]


class DynamicsNetworkConfig(BaseModel):
    layers: list[LayerConfig]


class PredictionNetworkConfig(BaseModel):
    layers: list[LayerConfig]


class NetworksConfig(BaseModel):
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
