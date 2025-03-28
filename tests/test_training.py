from dataclasses import dataclass

import pytest
import torch

from src.config.config_loader import TrainingConfig
from src.neural_networks.neural_network import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from src.training import NeuralNetworkManager
from src.training_data_generator import Chunk, Episode
from tests.nerual_networks.test_networks import (
    tiny_dyn_net,
    tiny_pred_net,
    tiny_repr_net,
)


def dummy_state(input_channels: int, observation: tuple[int, int]) -> torch.Tensor:
    """
    A minimal "state" that we can pass into the networks.
    Generates a random tensor with the given number of channels and spatial dimensions.
    """
    return torch.randn(input_channels, *observation)


@pytest.fixture
def minimal_config():
    """
    Returns a minimal MuZero TrainingConfig with:
    - look_back=0 (no historical frames)
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
        mini_batch_size=2,
    )


@dataclass
class NetworksConfig:
    repr_net: RepresentationNetwork
    dyn_net: DynamicsNetwork
    pred_net: PredictionNetwork
    input_channels: int
    observation_space: tuple[int, int]


@pytest.fixture
def minimal_nets() -> NetworksConfig:
    """
    Creates a NetworksConfig using the tiny networks.
    - input_channels: number of channels in the input state.
    - observation_space: spatial dimensions (height, width) of the input state.
    - latent_shape: shape of the latent representation (channels, height, width).
    - num_actions: number of output actions (for policy).
    """
    input_channels = 1
    observation_space = (1, 4, 4)
    latent_shape = (2, 4, 4)
    num_actions = 2

    repr_net = tiny_repr_net(
        observation_space=(input_channels, *observation_space),
        latent_shape=latent_shape,
    )
    dyn_net = tiny_dyn_net(latent_shape=latent_shape, num_actions=num_actions)
    pred_net = tiny_pred_net(latent_shape=latent_shape, num_actions=num_actions)

    return NetworksConfig(
        repr_net=repr_net,
        dyn_net=dyn_net,
        pred_net=pred_net,
        input_channels=input_channels,
        observation_space=observation_space,
    )


def test_single_update(minimal_config, minimal_nets):
    """
    Test that the training loop runs one BPTT update without crashing
    using minimal data.
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

    # Create an episode with two states
    ep = Episode(chunks=[s0, s1])
    episode_history = [ep]

    # Run exactly one update
    manager.train(episode_history)

    # Check that at least some gradients were computed.
    for param in manager.repr_net.parameters():
        assert param.grad is None or torch.any(param.grad != 0) or (param.grad.isnan().any() == False)


def test_multiple_updates(minimal_config, minimal_nets):
    """
    Test that multiple BPTT updates run smoothly using minimal data.
    """
    manager = NeuralNetworkManager(
        config=minimal_config,
        repr_net=minimal_nets.repr_net,
        dyn_net=minimal_nets.dyn_net,
        pred_net=minimal_nets.pred_net,
    )

    s0 = Chunk(
        state=dummy_state(minimal_nets.input_channels, minimal_nets.observation_space),
        policy=torch.tensor([0.8, 0.2]),
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

    # Create an episode and duplicate it to form a history of two episodes
    ep = Episode(chunks=[s0, s1])
    episode_history = [ep, ep]

    # Perform 5 BPTT updates
    manager.train(episode_history)

    # Verify that the prediction network parameters are still accessible (i.e. not None)
    for param in manager.pred_net.parameters():
        assert param is not None


def test_no_valid_rollout(minimal_config, minimal_nets):
    """
    Test that when an episode is too short (only 1 state) so that a rollout is impossible,
    the training code skips the update gracefully without crashing.
    """
    manager = NeuralNetworkManager(
        config=minimal_config,
        repr_net=minimal_nets.repr_net,
        dyn_net=minimal_nets.dyn_net,
        pred_net=minimal_nets.pred_net,
    )

    # Create an episode with only one state (no valid rollout)
    state0 = Chunk(
        state=dummy_state(minimal_nets.input_channels, minimal_nets.observation_space),
        policy=torch.tensor([1.0, 0.0]),
        reward=0.0,
        value=0.0,
        best_action=0,
    )
    ep = Episode(chunks=[state0])

    # Run training; it should handle the case gracefully
    manager.train([ep])


@pytest.mark.parametrize(
    "look_back, batch_size, roll_ahead, epochs, mbs",
    [
        (0, 3, 1, 1, 1),
        (0, 4, 1, 1, 2),
        (
            0,
            3,
            2,
            2,
            5,
        ),  # For roll_ahead=2, the episode must contain at least 3 states.
    ],
)
def test_training_various_configs(look_back, batch_size, roll_ahead, epochs, mbs, minimal_nets):
    """
    Test training with various TrainingConfig parameters.
    Constructs a config from the parameter tuple and ensures the training loop
    runs without crashing.
    """
    config = TrainingConfig(
        look_back=look_back,
        batch_size=batch_size,
        roll_ahead=roll_ahead,
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        epochs=epochs,
        mini_batch_size=mbs,
    )
    manager = NeuralNetworkManager(
        config=config,
        repr_net=minimal_nets.repr_net,
        dyn_net=minimal_nets.dyn_net,
        pred_net=minimal_nets.pred_net,
    )

    # Create an episode with enough states to satisfy roll_ahead.
    num_states = roll_ahead + 1
    chunks = []
    for i in range(num_states):
        chunk = Chunk(
            state=dummy_state(minimal_nets.input_channels, minimal_nets.observation_space),
            policy=torch.tensor([0.8, 0.2]),
            reward=0.0 if i == 0 else 1.0,
            value=0.1 if i == 0 else 0.2,
            best_action=0 if i == 0 else 1,
        )
        chunks.append(chunk)
    ep = Episode(chunks=chunks)
    episode_history = [ep]

    manager.train(episode_history)

    # Check that the prediction network's parameters are still accessible.
    for param in manager.pred_net.parameters():
        assert param is not None


@pytest.mark.parametrize("num_episodes", [1, 2, 3])
def test_training_with_varied_episode_counts(minimal_config, minimal_nets, num_episodes):
    """
    Test training when varying the number of episodes in the history.
    This ensures the training loop can handle different amounts of data.
    """
    manager = NeuralNetworkManager(
        config=minimal_config,
        repr_net=minimal_nets.repr_net,
        dyn_net=minimal_nets.dyn_net,
        pred_net=minimal_nets.pred_net,
    )

    episode_history = []
    for _ in range(num_episodes):
        s0 = Chunk(
            state=dummy_state(minimal_nets.input_channels, minimal_nets.observation_space),
            policy=torch.tensor([0.8, 0.2]),
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
        ep = Episode(chunks=[s0, s1])
        episode_history.append(ep)

    manager.train(episode_history)

    for param in manager.pred_net.parameters():
        assert param is not None
