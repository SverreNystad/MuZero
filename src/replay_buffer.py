import numpy as np

import wandb
from src.training_data_generator import Episode

# Set the random seed for reproducibility
np.random.seed(0)


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        alpha=0.6,
        beta: float = 1.0,
    ):
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
        self.beta = beta

        # Store (episode, priority)
        self.episode_buffer: list[tuple[Episode, float]] = []
        self.max_priority = 1.0

    def add_episodes(self, episodes: list[Episode]) -> None:
        """Insert new episodes at max priority so they have a chance to be sampled."""
        for ep in episodes:
            self.episode_buffer.append((ep, self.max_priority))

        # Cap size
        over_capacity = len(self.episode_buffer) - self.buffer_capacity
        if over_capacity > 0:
            self.episode_buffer = self.episode_buffer[over_capacity:]

        # Log the average amount of states in each episode
        avg_states = np.mean([len(ep.chunks) for ep, _ in self.episode_buffer])
        rewards = [
            sum(chunk.reward for chunk in ep.chunks) for ep, _ in self.episode_buffer
        ]  # [chunk.reward for ep, _ in self.episode_buffer for chunk in ep.chunks]
        if len(rewards) > 0:
            median_reward = np.median(rewards)
            max_reward = np.max(rewards)
        else:
            median_reward = 0
            max_reward = 0

        wandb.log(
            {
                "replay/average_states_per_episode": avg_states,
                "replay/total_episodes_in_buffer": len(self.episode_buffer),
                "replay/max_reward": max_reward,
                "replay/median_reward": median_reward,
            }
        )

    def sample_batch(self) -> tuple[list[Episode], list[int], list[float]]:
        """
        Sample a batch of episodes from the buffer.
        The sampling is done based on the priorities of the episodes.
        The higher the priority, the more likely the episode is to be sampled.

        Indices of the sampled episodes are also returned for updating priorities.

        Returns:
            batch(list[Episode]): Sampled episodes
            indices(list[int]): Indices of the sampled episodes in the buffer
            is_weights(list[float]): Importance Sampling weights for each episode
        """
        if not self.episode_buffer:
            return [], [], []

        # Collect and scale priorities
        priorities = [p for _, p in self.episode_buffer]
        scaled = [p**self.alpha for p in priorities]
        total = sum(scaled)
        if total < 1e-8:
            # fallback to uniform if all priorities are ~0
            probs = [1.0 / len(scaled)] * len(scaled)
        else:
            probs = [s / total for s in scaled]

        # Sample indices
        indices = np.random.choice(
            len(self.episode_buffer),
            size=min(self.batch_size, len(self.episode_buffer)),
            # With small buffers you may under‐sample high-priority episodes if you force them to be unique in a batch
            # We should allow duplicates of high-priority episodes
            replace=True,
            p=probs,
        )

        # Compute IS-weights
        N = len(self.episode_buffer)
        is_weights = [(N * probs[i]) ** (-self.beta) for i in indices]
        max_w = max(is_weights)
        is_weights = [w / max_w for w in is_weights]

        batch = [self.episode_buffer[i][0] for i in indices]

        sampling_entropy = -np.dot(probs, np.log(np.array(probs, dtype=np.float32) + 1e-8))
        wandb.log({"replay/sampling_entropy": sampling_entropy})

        return batch, list(indices), is_weights

    def update_priorities(self, indices: list[int], new_errors: list[float]) -> None:
        """
        After training, update the priorities of the sampled episodes.
        new_priority often = abs(td_error) + 1e-6
        """
        for idx, err in zip(indices, new_errors):
            updated = abs(err) + 1e-6
            self.max_priority = max(self.max_priority, updated)
            ep, _ = self.episode_buffer[idx]
            self.episode_buffer[idx] = (ep, updated)

    def __len__(self):
        return len(self.episode_buffer)
