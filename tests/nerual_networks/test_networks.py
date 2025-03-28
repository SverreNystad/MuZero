import random

import pytest
import torch

from src.config.config_loader import (
    ConvLayerConfig,
    DenseLayerConfig,
    DynamicsNetworkConfig,
    PredictionNetworkConfig,
    RepresentationNetworkConfig,
    ResBlockConfig,
)
from src.neural_networks.neural_network import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)


def tiny_representation_config() -> RepresentationNetworkConfig:
    """
    A minimal RepresentationNetworkConfig with no downsample layers,
    no residual blocks.
    """
    return RepresentationNetworkConfig(
        downsample=[
            ConvLayerConfig(out_channels=32, kernel_size=3, stride=2, padding=1),
        ],
        res_net=[
            ResBlockConfig(
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                activation="relu",
                pool_kernel_size=2,
                pool_stride=2,
            ),
        ],
    )


def tiny_dynamics_config() -> DynamicsNetworkConfig:
    """
    A minimal DynamicsNetworkConfig with no residual blocks,
    and a simple single-layer reward net.
    """
    return DynamicsNetworkConfig(
        res_net=[
            ResBlockConfig(
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                activation="relu",
                pool_kernel_size=0,
                pool_stride=0,
            )
        ],
        reward_net=[
            # Output 1 value for reward
            DenseLayerConfig(out_features=1, activation="none")
        ],
    )


def tiny_prediction_config(num_actions: int = 2) -> PredictionNetworkConfig:
    """
    A minimal PredictionNetworkConfig with no residual blocks,
    plus a single-layer value net and single-layer policy net.
    By default, policy_net out_features=2 (2 actions).
    """
    return PredictionNetworkConfig(
        res_net=[
            ResBlockConfig(
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                activation="relu",
                pool_kernel_size=0,
                pool_stride=0,
            )
        ],
        value_net=[
            # 1 output for value
            DenseLayerConfig(out_features=1, activation="none")
        ],
        policy_net=[
            # `num_actions` outputs for policy logits
            DenseLayerConfig(out_features=num_actions, activation="none")
        ],
    )


def tiny_repr_net(observation_space=(1, 1, 1), latent_shape=(2, 1, 1)):
    """
    A very small RepresentationNetwork:
    - Input shape: (1,1,1) => 1 channel, 1x1 image
    - Output latent_shape: (2,1,1) => 2 channels, 1x1
    """
    config = tiny_representation_config()
    return RepresentationNetwork(
        observation_space=observation_space,
        latent_shape=latent_shape,  # (C,H,W) for latent
        config=config,
    )


def tiny_dyn_net(latent_shape=(2, 1, 1), num_actions=2):
    config = tiny_dynamics_config()
    return DynamicsNetwork(
        latent_shape=latent_shape,
        num_actions=num_actions,
        config=config,
    )


def tiny_pred_net(latent_shape=(2, 1, 1), num_actions=2):
    config = tiny_prediction_config(num_actions=2)
    return PredictionNetwork(
        latent_shape=latent_shape,
        num_actions=num_actions,
        config=config,
    )


@pytest.mark.parametrize(
    "observation_space, latent_shape, num_actions",
    [
        ((3, 6, 6), (2, 6, 7), 10),
        ((1, 4, 4), (2, 4, 4), 10),
        ((1, 4, 3), (2, 2, 2), 10),
    ],
)
def test_minimal_forward_pass(observation_space, latent_shape, num_actions):
    """
    Test that the forward pass of the minimal networks runs without errors.
    """
    obs = torch.ones((1, *observation_space))
    repr_net = tiny_repr_net(latent_shape=latent_shape, observation_space=observation_space)
    dyn_net = tiny_dyn_net(latent_shape, num_actions=num_actions)
    pred_net = tiny_pred_net(latent_shape, num_actions=num_actions)
    latent = repr_net(obs)
    next_latent, reward = dyn_net(latent, torch.tensor([random.randint(0, num_actions)]))
    policy_logits, value = pred_net(latent)

    assert latent.shape == (1, *latent_shape)
    assert next_latent.shape == (1, *latent_shape)
    assert reward.shape == (1, 1)
    assert policy_logits.shape == (1, num_actions)
    assert value.shape == (1, 1)
