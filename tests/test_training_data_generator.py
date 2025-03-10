import os
import torch
from src.environments.car_racing import CarRacingConfig
from src.environments.factory import create_environment
from src.training_data_generator import (
    Episode,
    Chunk,
    TrainingDataGenerator,
    save_training_data,
    load_training_data,
    load_all_training_data,
)


def test_generate_training_data_gives_episode_data():
    env_config = CarRacingConfig(seed=42)
    env = create_environment(env_config)

    generator = TrainingDataGenerator(env)

    training_data = generator.generate_training_data(1, 30_000)

    assert len(training_data) == 1
    assert isinstance(training_data, list[Episode])


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
