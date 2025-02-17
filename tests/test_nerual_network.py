import pytest
import torch

from src.neural_network import (
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
    observation_space = (8, 8)
    latent_dim = 16

    # Create a RepresentationNetwork
    repr_net = RepresentationNetwork(
        input_channels=input_channels,
        latent_dim=latent_dim,
        observation_space=observation_space,
    )

    # Create a dummy observation
    # Shape: (batch_size, channels, height, width)
    observation = torch.randn(batch_size, input_channels, *observation_space)

    # Forward pass
    latent = repr_net(observation)

    # Check output shape: (batch_size, latent_dim)
    assert latent.shape == (batch_size, latent_dim), (
        f"RepresentationNetwork output shape {latent.shape} "
        f"does not match expected {(batch_size, latent_dim)}"
    )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_dynamics_network_forward(batch_size):
    """
    Test that the DynamicsNetwork receives a latent vector of correct shape
    and an integer tensor of action indices, and returns:
      - next_latent of shape (batch_size, latent_dim)
      - reward of shape (batch_size, 1)
    """
    latent_dim = 16
    num_actions = 5

    dyn_net = DynamicsNetwork(latent_dim=latent_dim, num_actions=num_actions)

    # Dummy latent state: shape (batch_size, latent_dim)
    latent = torch.randn(batch_size, latent_dim)

    # Dummy actions: integer tensor of shape (batch_size,)
    #   The network code uses one-hot encoding of these integers.
    actions = torch.randint(0, num_actions, (batch_size,))

    # Forward pass
    next_latent, reward = dyn_net(latent, actions)

    # Check shapes
    assert next_latent.shape == (batch_size, latent_dim), (
        f"DynamicsNetwork next_latent shape {next_latent.shape} "
        f"does not match expected {(batch_size, latent_dim)}"
    )
    assert reward.shape == (batch_size, 1), (
        f"DynamicsNetwork reward shape {reward.shape} "
        f"does not match expected {(batch_size, 1)}"
    )


@pytest.mark.parametrize("batch_size", [1, 4])
def test_prediction_network_forward(batch_size):
    """
    Test that the PredictionNetwork receives a latent vector of correct shape
    and returns:
      - policy_logits of shape (batch_size, num_actions)
      - value of shape (batch_size, 1)
    """
    latent_dim = 16
    num_actions = 5

    pred_net = PredictionNetwork(latent_dim=latent_dim, num_actions=num_actions)

    # Dummy latent state: shape (batch_size, latent_dim)
    latent = torch.randn(batch_size, latent_dim)

    # Forward pass
    policy_logits, value = pred_net(latent)

    # Check shapes
    assert policy_logits.shape == (batch_size, num_actions), (
        f"PredictionNetwork policy_logits shape {policy_logits.shape} "
        f"does not match expected {(batch_size, num_actions)}"
    )
    assert value.shape == (batch_size, 1), (
        f"PredictionNetwork value shape {value.shape} "
        f"does not match expected {(batch_size, 1)}"
    )
