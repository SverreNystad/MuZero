import pytest
import torch

from src.config.config_loader import (
    ConvLayerConfig,
    DenseLayerConfig,
    DynamicsNetworkConfig,
    PoolLayerConfig,
    PredictionNetworkConfig,
    RepresentationNetworkConfig,
    ResBlockConfig,
)
from src.nerual_networks.neural_network import (
    RepresentationNetwork,
    DynamicsNetwork,
    PredictionNetwork,
)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_representation_network_forward(batch_size):
    """
    Test that the RepresentationNetwork receives an observation
    of correct shape and returns the correct latent shape.
    """
    input_channels = 3
    observation_space = (input_channels, 96, 96)
    latent_shape = (input_channels, 6, 6)

    config = RepresentationNetworkConfig(
        downsample=[
            ConvLayerConfig(out_channels=32, kernel_size=3, stride=2, padding=1),
            ResBlockConfig(
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                activation="relu",
                pool_kernel_size=2,
                pool_stride=2,
            ),
            PoolLayerConfig(kernel_size=2, stride=2, pool_type="max"),
        ],
        res_net=[
            ResBlockConfig(
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                activation="relu",
                pool_kernel_size=0,
                pool_stride=0,
            )
        ],
    )
    # Create a RepresentationNetwork
    repr_net = RepresentationNetwork(
        latent_shape=latent_shape,
        observation_space=observation_space,
        config=config,
    )

    # Create a dummy observation
    # Shape: (batch_size, channels, height, width)
    observation = torch.randn(batch_size, *observation_space)

    # Forward pass
    latent = repr_net(observation)

    # Check output shape: (batch_size, latent_dim)
    assert latent.shape == (batch_size, *latent_shape), (
        f"RepresentationNetwork output shape {latent.shape} "
        f"does not match expected {(batch_size, *latent_shape)}"
    )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_dynamics_network_forward(batch_size):
    """
    Test that the DynamicsNetwork receives a latent tensor of shape
    (batch_size, C, H, W) and an integer tensor of action indices,
    then returns:
      - next_latent of shape (batch_size, C, H, W)
      - reward of shape (batch_size, 1)
    """

    # Suppose our latent is (C=8, H=4, W=4), so flattened = 8*4*4=128
    latent_shape = (8, 4, 4)
    action_space_size = 5

    # Minimal config ensuring the final output of res_net is 128,
    # so we can reshape back to (8,4,4).
    # And reward_net must produce 1 dimension for reward.
    dyn_config = DynamicsNetworkConfig(
        res_net=[
            DenseLayerConfig(out_features=128, activation="relu"),
        ],
        reward_net=[
            DenseLayerConfig(out_features=64, activation="relu"),
            DenseLayerConfig(out_features=1, activation="none"),
        ],
    )

    # Build the dynamics network
    dyn_net = DynamicsNetwork(latent_shape, action_space_size, dyn_config)

    # Dummy latent state: shape (batch_size, C, H, W)
    latent = torch.randn(batch_size, *latent_shape)

    # Dummy integer actions: shape (batch_size,)
    actions = torch.randint(0, action_space_size, (batch_size,))

    # Forward pass
    next_latent, reward = dyn_net(latent, actions)

    # Check shapes
    assert next_latent.shape == (batch_size, *latent_shape), (
        f"DynamicsNetwork next_latent shape {next_latent.shape} "
        f"does not match expected {(batch_size, *latent_shape)}"
    )
    assert reward.shape == (batch_size, 1), (
        f"DynamicsNetwork reward shape {reward.shape} "
        f"does not match expected {(batch_size, 1)}"
    )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_prediction_network_forward(batch_size):
    """
    Test that the PredictionNetwork receives a latent tensor of shape
    (batch_size, C, H, W) and returns:
      - policy_logits of shape (batch_size, num_actions)
      - value of shape (batch_size, 1)
    """

    # Suppose our latent is (C=8, H=4, W=4)
    latent_shape = (8, 4, 4)
    action_space_size = 5

    # We'll create a PredictionNetworkConfig that:
    # 1) Has a small trunk (res_net) that outputs 64,
    # 2) A value_net that ends in out_features=1,
    # 3) A policy_net that ends in out_features=action_space_size
    pred_config = PredictionNetworkConfig(
        res_net=[
            DenseLayerConfig(out_features=64, activation="relu"),
        ],
        value_net=[
            DenseLayerConfig(out_features=1, activation="none"),
        ],
        policy_net=[
            DenseLayerConfig(out_features=action_space_size, activation="none"),
        ],
    )

    pred_net = PredictionNetwork(latent_shape, action_space_size, pred_config)

    # Dummy latent state: shape (batch_size, C, H, W)
    latent = torch.randn(batch_size, *latent_shape)

    # Forward pass
    policy_logits, value = pred_net(latent)

    # Check shapes
    assert policy_logits.shape == (batch_size, action_space_size), (
        f"PredictionNetwork policy_logits shape {policy_logits.shape} "
        f"does not match expected {(batch_size, action_space_size)}"
    )
    assert value.shape == (batch_size, 1), (
        f"PredictionNetwork value shape {value.shape} "
        f"does not match expected {(batch_size, 1)}"
    )
