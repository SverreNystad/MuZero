import pytest
import torch

from src.replay_buffer import ReplayBuffer
from src.training_data_generator import Chunk, Episode


@pytest.mark.parametrize(
    "buffer_size, batch_size",
    [
        (0, 2),  # Invalid buffer size
        (5, 0),  # Invalid batch size
        (2, 5),  # Batch size exceeds buffer size
        (-1, 2),  # Negative buffer size
        (2, -1),  # Negative batch size
    ],
)
def test_batch_size_exceeds_buffer(buffer_size, batch_size):
    with pytest.raises(ValueError):
        ReplayBuffer(buffer_size, batch_size)


def test_adding_episode_to_replay_buffer():
    # Create a ReplayBuffer instance
    window_size = 1
    batch_size = 1
    buffer = ReplayBuffer(window_size, batch_size)

    # Create a sample episode with dummy data
    episode = Episode(
        chunks=[
            Chunk(
                state=torch.tensor([0.0, 1.0]),
                policy=torch.tensor([0.5, 0.5]),
                reward=1.0,
                value=0.5,
                best_action=1,
            )
        ]
    )
    # Add the episode to the buffer
    buffer.add_episodes([episode])
    assert len(buffer.episode_buffer) == 1


def test_adding_more_games_then_window_size():
    # Create a ReplayBuffer instance
    buffer_size = 1
    batch_size = 1
    buffer = ReplayBuffer(buffer_size, batch_size)

    # Create a sample episode with dummy data
    episode1 = Episode(
        chunks=[
            Chunk(
                state=torch.tensor([0.0, 1.0]),
                policy=torch.tensor([0.5, 0.5]),
                reward=1.0,
                value=0.5,
                best_action=1,
            )
        ]
    )
    episode2 = Episode(
        chunks=[
            Chunk(
                state=torch.tensor([1.0, 0.0]),
                policy=torch.tensor([0.5, 0.5]),
                reward=1.0,
                value=0.5,
                best_action=1,
            )
        ]
    )
    # Add the episode to the buffer

    buffer.add_episodes([episode1, episode2])

    # Check if the buffer has only the last episode
    assert len(buffer) == 1
    batch, indices = buffer.sample_batch()
    assert len(batch) == 1
    assert batch[0] == episode2


def test_sample_batch_from_replay_buffer():
    # Create a ReplayBuffer instance
    buffer_size = 10
    batch_size = 2
    buffer = ReplayBuffer(buffer_size, batch_size)

    # Create sample episodes with dummy data
    episode1 = Episode(
        chunks=[
            Chunk(
                state=torch.tensor([0.0, 1.0]),
                policy=torch.tensor([0.5, 0.5]),
                reward=1.0,
                value=0.5,
                best_action=1,
            )
        ]
    )
    episode2 = Episode(
        chunks=[
            Chunk(
                state=torch.tensor([1.0, 0.0]),
                policy=torch.tensor([0.5, 0.5]),
                reward=1.0,
                value=0.5,
                best_action=1,
            )
        ]
    )
    # Add the episodes to the buffer
    buffer.add_episodes([episode1, episode2])

    # Sample a batch from the buffer
    batch, indices = buffer.sample_batch()

    # Check if the batch size is correct
    assert len(batch) == batch_size


def test_sample_batch_with_less_than_batch_size():
    # Create a ReplayBuffer instance
    buffer_size = 10
    batch_size = 5
    buffer = ReplayBuffer(buffer_size, batch_size)

    # Create sample episodes with dummy data
    episode1 = Episode(
        chunks=[
            Chunk(
                state=torch.tensor([0.0, 1.0]),
                policy=torch.tensor([0.5, 0.5]),
                reward=1.0,
                value=0.5,
                best_action=1,
            )
        ]
    )
    episode2 = Episode(
        chunks=[
            Chunk(
                state=torch.tensor([1.0, 0.0]),
                policy=torch.tensor([0.5, 0.5]),
                reward=1.0,
                value=0.5,
                best_action=1,
            )
        ]
    )
    # Add the episodes to the buffer
    episode_history = [episode1, episode2]
    buffer.add_episodes(episode_history)

    # Sample a batch from the buffer
    batch, indices = buffer.sample_batch()

    # Check if the batch size is correct
    assert len(batch) == len(episode_history)


def test_sample_batch_with_no_episodes():
    # Create a ReplayBuffer instance
    buffer_size = 10
    batch_size = 5
    buffer = ReplayBuffer(buffer_size, batch_size)

    # Sample a batch from the empty buffer
    batch, indices = buffer.sample_batch()

    # Check if the batch is empty
    assert len(batch) == 0
