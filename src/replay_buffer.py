import random

from src.training_data_generator import Episode


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int):
        """
        Initializes the ReplayBuffer with an empty list of episodes.
        """
        if buffer_size <= 0:
            raise ValueError("Buffer size must be greater than 0.")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        if buffer_size < batch_size:
            raise ValueError("Buffer size must be greater than batch size.")

        self.episode_buffer = []
        self.buffer_capacity = buffer_size
        self.batch_size = batch_size

    def add_episodes(self, episodes: list[Episode]) -> None:
        self.episode_buffer.extend(episodes)
        over_capacity = len(self.episode_buffer) - self.buffer_capacity
        if over_capacity > 0:
            # Remove the oldest episodes to make space
            self.episode_buffer = self.episode_buffer[over_capacity:]

    def sample_batch(self) -> list[Episode]:
        """
        Samples a batch of episodes from the buffer.

        Returns:
            list[Episode]: A list of sampled episodes.
        """
        return random.sample(self.episode_buffer, self.batch_size)
