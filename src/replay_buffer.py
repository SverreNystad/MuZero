import numpy as np

from src.training_data_generator import Episode

# Set the random seed for reproducibility
np.random.seed(0)


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, alpha=0.6):
        """
        Prioritized Replay Buffer.
        Args:
            buffer_size(int): Max number of episodes to store
            batch_size(int): Sample size
            alpha(float): Exponent for how strongly to use priorities (0=uniform, 1=full priority)
        """
        if buffer_size <= 0:
            raise ValueError("Buffer size must be greater than 0.")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        if buffer_size < batch_size:
            raise ValueError("Buffer size must be greater than batch size.")

        self.buffer_capacity = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha

        # Store (episode, priority)
        self.episode_buffer = []
        self.max_priority = 1.0

    def add_episodes(self, episodes: list[Episode]) -> None:
        """Insert new episodes at max priority so they have a chance to be sampled."""
        for ep in episodes:
            self.episode_buffer.append((ep, self.max_priority))

        # Cap size
        over_capacity = len(self.episode_buffer) - self.buffer_capacity
        if over_capacity > 0:
            self.episode_buffer = self.episode_buffer[over_capacity:]

    def sample_batch(self) -> tuple[list[Episode], list[int]]:
        """
        Sample a batch of episodes from the buffer.
        The sampling is done based on the priorities of the episodes.
        The higher the priority, the more likely the episode is to be sampled.

        Indices of the sampled episodes are also returned for updating priorities.

        Returns:
            batch(list[Episode]): Sampled episodes
            indices(list[int]): Indices of the sampled episodes in the buffer
        """
        if not self.episode_buffer:
            return ([], [])

        # Collect priorities
        priorities = [p for _, p in self.episode_buffer]
        scaled = [p**self.alpha for p in priorities]

        # Build distribution & sample
        total = sum(scaled)
        if total < 1e-8:
            # fallback to uniform if all priorities are ~0
            probs = [1.0 / len(scaled)] * len(scaled)
        else:
            probs = [v / total for v in scaled]

        indices = np.random.choice(
            len(self.episode_buffer),
            size=min(self.batch_size, len(self.episode_buffer)),
            replace=False,
            p=probs,
        )
        batch = []
        for idx in indices:
            batch.append(self.episode_buffer[idx][0])
        return batch, indices

    def update_priorities(self, indices: list[int], new_errors: list[float]) -> None:
        """
        After training, update the priorities of the sampled episodes.
        new_priority often = abs(td_error) + 1e-6
        """
        for idx, err in zip(indices, new_errors):
            updated = abs(err) + 1e-6
            if updated > self.max_priority:
                self.max_priority = updated
            ep, _ = self.episode_buffer[idx]
            self.episode_buffer[idx] = (ep, updated)

    def __len__(self):
        return len(self.episode_buffer)
