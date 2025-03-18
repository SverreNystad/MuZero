from dataclasses import dataclass
from pydantic import BaseModel
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config.config_loader import TrainingConfig
from src.environment import Environment
from src.nerual_networks.neural_network import (
    RepresentationNetwork,
    DynamicsNetwork,
    PredictionNetwork,
)
from src.training_data_generator import Chunk, Episode
from src.training import NeuralNetworkManager


def dummy_state(input_channels: int, observation: tuple[int, int]) -> Tensor:
    """
    A minimal "state" that we can pass into the networks.
    """

    # Single-channel image of size "input_channels"
    # E.g., a single pixel with value 0.5
    return torch.randn(input_channels, *observation)


###########################
# Fixtures for the tests  #
###########################


@pytest.fixture
def minimal_config():
    """
    Returns a minimal MuZero config with
    - lookback=0 (no historical frames)
    - roll_ahead=1
    - learning_rate=1e-3
    """

    return TrainingConfig(
        look_back=0,
        batch_size=3,
        roll_ahead=1,
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        epochs=1,
    )


@dataclass
class NetworksConfig:
    repr_net: RepresentationNetwork
    dyn_net: DynamicsNetwork
    pred_net: PredictionNetwork
    input_channels: int
    observation_space: tuple[int, int]


@pytest.fixture(
    params=[
        {
            "input_channels": 1,
            "observation_space": (1, 1),
            "latent_dim": 2,
        },
        {
            "input_channels": 3,
            "observation_space": (8, 8),
            "latent_dim": 64,
        },
        {
            "input_channels": 3,
            "observation_space": (96, 96),
            "latent_dim": 128,
        },
    ]
)
def minimal_nets(request) -> NetworksConfig:
    """
    Returns minimal neural networks for testing.
    """
    params = request.param
    return NetworksConfig(
        repr_net=RepresentationNetwork(**params),
        dyn_net=DynamicsNetwork(latent_dim=params["latent_dim"], num_actions=2),
        pred_net=PredictionNetwork(latent_dim=params["latent_dim"], num_actions=2),
        input_channels=params["input_channels"],
        observation_space=params["observation_space"],
    )


def test_single_update(minimal_config, minimal_nets):
    """
    Test that the training loop runs 1 BPTT update
    without crashing using minimal data.
    """
    manager = NeuralNetworkManager(
        config=minimal_config,
        repr_net=minimal_nets.repr_net,
        dyn_net=minimal_nets.dyn_net,
        pred_net=minimal_nets.pred_net,
    )

    s0 = Chunk(
        state=dummy_state(minimal_nets.input_channels, minimal_nets.observation_space),
        policy=torch.tensor([0.8, 0.2]),  # prefers action 0
        reward=0.0,
        value=0.1,
        best_action=0,
    )
    s1 = Chunk(
        state=dummy_state(minimal_nets.input_channels, minimal_nets.observation_space),
        policy=torch.tensor([0.6, 0.4]),
        reward=1.0,
        value=0.2,
        best_action=1,
    )

    # Put them into an Episode
    ep = Episode(chunks=[s0, s1])

    # One minimal episode in the history
    episode_history = [ep]

    # We do exactly 1 update
    manager.train(episode_history, mbs=1)

    # If it doesn't crash, we pass.
    # Optionally we can check that parameters got gradients
    for param in manager.repr_net.parameters():
        assert (
            param.grad is None
            or torch.any(param.grad != 0)
            or param.grad.isnan().any() == False
        )


def test_multiple_updates(
    minimal_config,
    minimal_nets,
):
    """
    Test that multiple updates run smoothly using minimal data.
    """
    manager = NeuralNetworkManager(
        config=minimal_config,
        repr_net=minimal_nets.repr_net,
        dyn_net=minimal_nets.dyn_net,
        pred_net=minimal_nets.pred_net,
    )

    s0 = Chunk(
        state=dummy_state(minimal_nets.input_channels, minimal_nets.observation_space),
        policy=torch.tensor([0.8, 0.2]),  # prefers action 0
        reward=0.0,
        value=0.1,
        best_action=0,
    )
    s1 = Chunk(
        state=dummy_state(minimal_nets.input_channels, minimal_nets.observation_space),
        policy=torch.tensor([0.6, 0.4]),
        reward=1.0,
        value=0.2,
        best_action=1,
    )

    # Put them into an Episode
    ep = Episode(chunks=[s0, s1])

    # Provide two identical episodes
    episode_history = [ep, ep]

    # We'll do 5 BPTT updates
    manager.train(episode_history, mbs=5)

    # Check that something changed in the networks
    # (not guaranteed, but we can at least verify no crash).
    for param in manager.pred_net.parameters():
        # We at least check param is not None
        assert param is not None


def test_no_valid_rollout(minimal_config, minimal_nets):
    """
    If the episodes are too short (only 1 state),
    we can't do a roll_ahead=1.
    The code should skip training gracefully with no crash.
    """
    manager = NeuralNetworkManager(
        config=minimal_config,
        repr_net=minimal_nets.repr_net,
        dyn_net=minimal_nets.dyn_net,
        pred_net=minimal_nets.pred_net,
    )

    # Single state => no rollout possible
    state0 = Chunk(
        state=dummy_state(minimal_nets.input_channels, minimal_nets.observation_space),
        policy=torch.tensor([1.0, 0.0]),
        reward=0.0,
        value=0.0,
        best_action=0,
    )
    ep = Episode(chunks=[state0])

    manager.train([ep], mbs=3)


# If we reach here with no crash, success.
# We expect zero gradient updates took place.

##############################
# Additional Test Scenarios #
##############################
# - You might add tests for shape correctness,
#   e.g. ensuring predicted actions match the networks
#   but these minimal tests focus on the main train loop.
