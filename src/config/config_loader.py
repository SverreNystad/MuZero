import os
from enum import StrEnum
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, Field

from src.environments.car_racing import CarRacingConfig
from src.environments.connect_four import ConnectFourConfig
from src.environments.flappy_bird import FlappyBirdConfig
from src.environments.lunar_lander import LunarLanderConfig

CONFIG_PATH = os.path.dirname(__file__)


EnvironmentConfig = Union[CarRacingConfig, ConnectFourConfig, LunarLanderConfig, FlappyBirdConfig]


class SelectionStrategyType(StrEnum):
    uct = "uct"
    puct = "puct"


class MCTSConfig(BaseModel):
    selection_strategy: SelectionStrategyType = SelectionStrategyType.puct
    max_iterations: int
    max_time: int
    model_look_ahead: int = 5
    discount_factor: float = 1.0
    dirichlet_alpha: float = 0.3
    noise_frac: float = 0.25
    visualize: bool = False


class ConvLayerConfig(BaseModel):
    type: Literal["conv_layer"] = "conv_layer"
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    activation: str | None = "relu"  # e.g. "relu", "tanh", "sigmoid", "none"


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
    out_features: int | None = None  # for conv or linear layers
    activation: str | None = "relu"  # e.g. "relu", "tanh", "sigmoid", "none"


DownsampleLayerConfig = Annotated[Union[ConvLayerConfig, PoolLayerConfig, ResBlockConfig], Field(discriminator="type")]


class RepresentationNetworkConfig(BaseModel):
    history_length: int
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
    # Epsilon greedy action selection aka (exploration)
    epsilon: float
    epsilon_decay: float
    random_action_probability: list[float] | None = None
    min_epsilon: float = 0.25  # MuZero has a minimum random action probability of 0.25


class ValidationConfig(BaseModel):
    video_upload_interval: int
    simulation_count: int
    simulation_depth: int


class TrainingConfig(BaseModel):
    learning_rate: float
    weight_decay: float
    batch_size: int
    replay_buffer_size: int
    alpha: float  # priority exponent
    discount_factor: float
    epochs: int
    betas: tuple[float, float]
    roll_ahead: int
    look_back: int
    mini_batch_size: int
    reward_coefficient: float
    value_coefficient: float
    policy_coefficient: float
    min_learning_rate: float
    total_training_steps: int
    lr_schedule: str  # e.g. "linear", "cosine", "step"
    optimizer: str = "sgd"  # e.g.  "sgd", "adam", "adamw","rmsprop"
    momentum: float = 0.9
    # scheduler_milestones = Field(default=[8000, 20000])
    scheduler_gamma: float = 0.99971
    scheduler_T_max: int = 40000  # total optimiser steps you expect
    scheduler_eta_min: float = 1e-5  # final LR (10-4 × LR₀)


class RunTimeConfig(BaseModel):
    use_cuda: bool


class Configuration(BaseModel):
    environment: EnvironmentConfig = Field(..., discriminator="type")
    networks: NetworksConfig
    training_data_generator: TrainingDataGeneratorConfig
    training: TrainingConfig
    validation: ValidationConfig
    runtime: RunTimeConfig
    project_name: str = "muzero"


def load_config(filename: str) -> Configuration:
    """
    Load a configuration file from the config directory.
    """
    path = os.path.join(CONFIG_PATH, filename)
    with open(path) as file:
        raw_config = yaml.safe_load(file)
    return Configuration(**raw_config)
