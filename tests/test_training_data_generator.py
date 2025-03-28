import os

import pytest
import torch

from src.config.config_loader import TrainingDataGeneratorConfig
from src.environments.car_racing import CarRacingConfig
from src.environments.factory import create_environment
from src.search.factory import MCTSConfig
from src.training_data_generator import (
    Chunk,
    Episode,
    TrainingDataGenerator,
    load_all_training_data,
    load_training_data,
    save_training_data,
)
from tests.nerual_networks.test_networks import (
    tiny_dyn_net,
    tiny_pred_net,
    tiny_repr_net,
)


@pytest.fixture
def car_racing_env():
    """
    Pytest fixture that creates and returns a CarRacing environment with a fixed seed.
    """
    env_config = CarRacingConfig(seed=42)
    return create_environment(env_config)


def test_generate_training_data_gives_episode_data(car_racing_env):
    """
    Test that TrainingDataGenerator properly creates a list of Episode objects
    with at least one Chunk each.
    """
    latent_shape = (2, 2, 2)
    rep_net = tiny_repr_net(
        latent_shape=latent_shape,
        observation_space=car_racing_env.get_observation_space(),
    )
    dyn_net = tiny_dyn_net(latent_shape, num_actions=5)
    pred_net = tiny_pred_net(latent_shape, num_actions=5)

    # Example config for generating a small amount of data.
    mcts_config = MCTSConfig(
        max_iterations=1,
        max_time=1,
    )
    config = TrainingDataGeneratorConfig(
        num_episodes=1,
        max_steps_per_episode=10,
        look_back=3,
        roll_ahead=3,
        total_time=10,
        mcts=mcts_config,
    )

    generator = TrainingDataGenerator(
        env=car_racing_env,
        repr_net=rep_net,
        dyn_net=dyn_net,
        pred_net=pred_net,
        config=config,
    )

    training_data = generator.generate_training_data()
    # Check basic structure
    assert isinstance(training_data, list)
    assert len(training_data) == 1, "Expected exactly 1 episode."

    episode = training_data[0]
    assert isinstance(episode, Episode), "Expected Episode type."
    assert len(episode.chunks) > 0, "Episode should contain at least one chunk."

    chunk = episode.chunks[0]
    assert isinstance(chunk, Chunk), "Expected Chunk type."
    assert hasattr(chunk, "state"), "Chunk should have a 'state' attribute."
    assert hasattr(chunk, "policy"), "Chunk should have a 'policy' attribute."
    assert hasattr(chunk, "reward"), "Chunk should have a 'reward' attribute."
    assert hasattr(chunk, "value"), "Chunk should have a 'value' attribute."
    assert hasattr(chunk, "best_action"), "Chunk should have a 'best_action' attribute."


def test_save_and_load_episodes():
    chunk_1 = Chunk(
        state=torch.tensor([1]),
        policy=torch.tensor([0.5]),
        reward=0.0,
        value=0.0,
        best_action=torch.tensor([0]),
    )
    chunk_2 = Chunk(
        state=torch.tensor([2]),
        policy=torch.tensor([0.5]),
        reward=0.0,
        value=0.0,
        best_action=torch.tensor([0]),
    )
    chunk_3 = Chunk(
        state=torch.tensor([3]),
        policy=torch.tensor([0.5]),
        reward=0.0,
        value=0.0,
        best_action=torch.tensor([0]),
    )
    episodes = [
        Episode(chunks=[chunk_1, chunk_2]),
        Episode(chunks=[chunk_3]),
    ]

    path = save_training_data(episodes)
    loaded_episodes = load_training_data(path)

    assert episodes == loaded_episodes

    # Clean up
    if os.path.exists(path):
        os.remove(path)


def test_load_all_episodes():
    chunk_1 = Chunk(
        state=torch.tensor([1]),
        policy=torch.tensor([0.5]),
        reward=0.0,
        value=0.0,
        best_action=torch.tensor([0]),
    )
    chunk_2 = Chunk(
        state=torch.tensor([2]),
        policy=torch.tensor([0.5]),
        reward=0.0,
        value=0.0,
        best_action=torch.tensor([0]),
    )
    chunk_3 = Chunk(
        state=torch.tensor([3]),
        policy=torch.tensor([0.5]),
        reward=0.0,
        value=0.0,
        best_action=torch.tensor([0]),
    )
    episodes = [
        Episode(chunks=[chunk_1, chunk_2]),
        Episode(chunks=[chunk_3]),
    ]
    # Create some episodes
    path_1 = save_training_data(episodes)
    path_2 = save_training_data(episodes)

    loaded_episodes = load_all_training_data()
    assert len(loaded_episodes) >= len(episodes) * 2
    for episode in episodes:
        assert episode in loaded_episodes

    # Clean up
    if os.path.exists(path_1):
        os.remove(path_1)
    if os.path.exists(path_2):
        os.remove(path_2)
