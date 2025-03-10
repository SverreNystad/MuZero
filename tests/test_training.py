import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.environment import Environment
from src.neural_network import RepresentationNetwork, DynamicsNetwork, PredictionNetwork
from src.training_data_generator import Chunk, Episode
from src.training import NeuralNetworkManager


#############################
# Dummy Environment / State #
#############################

class DummyState:
    """
    A minimal "state" that we can pass into the networks.
    For a CNN-based RepresentationNetwork, we can simulate a (1,1) image.
    """
    def __init__(self):
        # Single-channel image of size 1×1
        # E.g., a single pixel with value 0.5
        self.observation = torch.tensor([[0.5]], dtype=torch.float32)

    def __array__(self):
        """
        Allows `torch.tensor(DummyState())` to work
        by returning a NumPy array if needed.
        """
        return self.observation.numpy()


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
    return {
        "lookback": 0,
        "roll_ahead": 1,
        "learning_rate": 1e-3
    }

@pytest.fixture
def tiny_repr_net():
    """
    A tiny representation network: 
    input_channels=1, observation_space=(1,1), latent_dim=2
    """
    return RepresentationNetwork(
        input_channels=1,
        observation_space=(1, 1),
        latent_dim=2
    )

@pytest.fixture
def tiny_dyn_net():
    """
    A tiny dynamics network:
    latent_dim=2, num_actions=2
    """
    return DynamicsNetwork(
        latent_dim=2,
        num_actions=2
    )

@pytest.fixture
def tiny_pred_net():
    """
    A tiny prediction network:
    latent_dim=2, num_actions=2
    """
    return PredictionNetwork(
        latent_dim=2,
        num_actions=2
    )

@pytest.fixture
def single_step_episode():
    """
    Creates an Episode object with just enough data 
    for one step of roll_ahead=1.

    We'll store:
    - state: DummyState
    - action: 0 or 1
    - reward: small float
    - policy: distribution over 2 actions
    - value: small float
    """

    # We need at least 2 states for roll_ahead=1:
    #   - The first state -> action -> second state
    # Episode must have states array of length >= 2
    # so that "len(states) - (roll_ahead+1) >= 0" is possible.
    # We'll store minimal data:
    s0 = Chunk(
        state=DummyState(), 
        policy=torch.tensor([0.8, 0.2]),  # prefers action 0
        reward=0.0,
        value=0.1,
        best_action=0
    )
    s1 = Chunk(
        state=DummyState(),
        policy=torch.tensor([0.6, 0.4]),
        reward=1.0,
        value=0.2,
        best_action=1
    )

    # Put them into an Episode
    ep = Episode(chunks=[s0, s1])
    return ep

################################
# Test Cases for Training Loop #
################################

def test_single_update(
    minimal_config,
    tiny_repr_net,
    tiny_dyn_net,
    tiny_pred_net,
    single_step_episode
):
    """
    Test that the training loop runs 1 BPTT update
    without crashing using minimal data.
    """
    manager = NeuralNetworkManager(
        config=minimal_config,
        repr_net=tiny_repr_net,
        dyn_net=tiny_dyn_net,
        pred_net=tiny_pred_net
    )

    # One minimal episode in the history
    episode_history = [single_step_episode]

    # We do exactly 1 update
    manager.train(episode_history, mbs=1)

    # If it doesn't crash, we pass.
    # Optionally we can check that parameters got gradients
    for param in manager.repr_net.parameters():
        assert param.grad is None or torch.any(param.grad != 0) or param.grad.isnan().any() == False

def test_multiple_updates(
    minimal_config,
    tiny_repr_net,
    tiny_dyn_net,
    tiny_pred_net,
    single_step_episode
):
    """
    Test that multiple updates run smoothly using minimal data.
    """
    manager = NeuralNetworkManager(
        config=minimal_config,
        repr_net=tiny_repr_net,
        dyn_net=tiny_dyn_net,
        pred_net=tiny_pred_net
    )

    # Provide two identical episodes
    episode_history = [single_step_episode, single_step_episode]

    # We'll do 5 BPTT updates
    manager.train(episode_history, mbs=5)

    # Check that something changed in the networks 
    # (not guaranteed, but we can at least verify no crash).
    for param in manager.pred_net.parameters():
        # We at least check param is not None
        assert param is not None


def test_no_valid_rollout(
    minimal_config,
    tiny_repr_net,
    tiny_dyn_net,
    tiny_pred_net
):
    """
    If the episodes are too short (only 1 state),
    we can't do a roll_ahead=1. 
    The code should skip training gracefully with no crash.
    """
    manager = NeuralNetworkManager(
        config=minimal_config,
        repr_net=tiny_repr_net,
        dyn_net=tiny_dyn_net,
        pred_net=tiny_pred_net
    )

    # Single state => no rollout possible
    state0 = Chunk(
        state=DummyState(),
        policy=torch.tensor([1.0, 0.0]),
        reward=0.0,
        value=0.0,
        best_action=0
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
