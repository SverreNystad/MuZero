from dataclasses import dataclass

import numpy as np
import torch

import wandb
from src.training_data_generator import Episode


@dataclass
class _Index:
    game_id: int  # episode index in self._games
    pos: int  # chunk index inside that episode


class ReplayBuffer:
    """
    Two–level Prioritised Experience Replay (games, then chunks)
    ------------------------------------------------------------
    * every chunk has its own priority p_i >= ε
    * a game priority is max(p_i) inside that game
    * sampling:   P(game)=p_game^α / Σ;  P(pos|game)=p_i^α / Σ;
    * importance-weights w_i ∝ (1/N·P(game)·P(pos|game))^β
    """

    def __init__(
        self,
        max_episodes: int,
        max_steps: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 500_000,
        eps: float = 1e-6,
    ):
        self.max_episodes = max_episodes
        self.max_steps = max_steps  # used for β-annealing
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps

        self._games: list[Episode] = []
        self._priorities: list[np.ndarray] = []  # one 1-D array per game
        self._game_p: list[float] = []  # max priority per game
        self._frame = 0  # global counter for β

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #

    def add_episodes(self, episodes: list[Episode]) -> None:
        """Insert a new game; new chunks get max priority so they are seen once."""
        for episode in episodes:
            prios = np.ones(len(episode.chunks), dtype=np.float32)
            if self._game_p:
                prios *= max(self._game_p)  # current max
            self._insert(episode, prios)
            self._trim()

        # Log the average amount of states in each episode
        avg_states = np.mean([len(ep.chunks) for ep in self._games])
        rewards = [
            sum(chunk.reward for chunk in ep.chunks) for ep in self._games
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
                "replay/total_episodes_in_buffer": len(self._games),
                "replay/max_reward": max_reward,
                "replay/median_reward": median_reward,
            }
        )

    def sample_batch(self, batch_size: int):
        """Return lists (episodes, positions, weights) ready for training."""
        if not self._games:
            return [], [], torch.tensor([])

        beta = self._annealed_beta()
        # --- sample games --------------------------------------------------
        game_probs = self._scaled(np.array(self._game_p))
        game_indices = np.random.choice(len(self._games), batch_size, p=game_probs)
        # --- sample positions inside each game -----------------------------
        samples: list[tuple[Episode, int, float]] = []
        for g in game_indices:
            pos_probs = self._scaled(self._priorities[g])
            pos = np.random.choice(len(pos_probs), p=pos_probs)
            p_sample = game_probs[g] * pos_probs[pos]
            weight = (1.0 / (len(self) * p_sample)) ** beta
            samples.append((self._games[g], pos, weight))

        # normalise weights
        weights = torch.tensor([w for *_, w in samples], dtype=torch.float32)
        weights /= weights.max()

        episodes = [e for (e, _, _) in samples]
        positions = [p for (_, p, _) in samples]

        return episodes, positions, weights, list(game_indices)

    def update_priorities(self, batch_games: list[int], batch_pos: list[int], td_errors: np.ndarray) -> None:
        """Write back absolute TD-errors."""
        for g, p, err in zip(batch_games, batch_pos, td_errors):
            prio = abs(err) + self.eps
            self._priorities[g][p] = prio
            self._game_p[g] = self._priorities[g].max()

    def __len__(self):
        return sum(map(len, self._priorities))

    # --------------------------------------------------------------------- #
    # internal helpers
    # --------------------------------------------------------------------- #
    def _insert(self, ep: Episode, prios: np.ndarray):
        self._games.append(ep)
        self._priorities.append(prios)
        self._game_p.append(prios.max())

    def _trim(self):
        while len(self._games) > self.max_episodes:
            self._games.pop(0)
            self._priorities.pop(0)
            self._game_p.pop(0)

    def _scaled(self, x: np.ndarray):
        scaled = x**self.alpha
        return scaled / scaled.sum()

    def _annealed_beta(self):
        self._frame += 1
        return min(1.0, self.beta_start + (1 - self.beta_start) * self._frame / self.beta_frames)
