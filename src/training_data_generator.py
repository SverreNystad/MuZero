from tqdm import trange
from torch import Tensor
from dataclasses import dataclass
from src.environment import Environment
from src.neural_network import RepresentationNetwork, PredictionNetwork, DynamicsNetwork


@dataclass
class Chunk:
    """
    Data class to store chunk state data.
    """

    state: Tensor
    policy: Tensor
    reward: float
    value: float  # TODO: Could be a Tensor with shape (1, 1)
    best_action: Tensor  # TODO: Could be an int


@dataclass
class Episode:
    """
    Data class to store episode data.
    """

    chunks: list[Chunk]


class TrainingDataGenerator:
    def __init__(
        self,
        env: Environment,
        mcts: callable,
        repr_net: RepresentationNetwork,
        dyn_net: DynamicsNetwork,
        pred_net: PredictionNetwork,
        config: dict,
    ):
        self.env = env
        self.mcts = mcts
        self.repr_net = repr_net
        self.dyn_net = dyn_net
        self.pred_net = pred_net
        self.num_episodes: int = config["num_episodes"]  # N_e
        self.max_steps: int = config["max_steps"]  # N_es
        self.look_back: int = config["look_back"]  # q
        self.total_time: float = config["total_time"]

    def generate_training_data(self) -> list[Episode]:
        """
        Args:
            num_episodes: Number of episodes to run.
            total_time: Total time to run the episodes in milliseconds (ms).

        Returns:
            List of episode data.
        """
        episode_history: list[Episode] = []

        for _ in trange(self.num_episodes):
            self.env.reset()

            epidata = Episode(chunks=[])
            state = self.env.get_state()
            for _ in trange(self.max_steps):
                latent_state = self.repr_net(state)

                tree_policy: Tensor
                tree_policy, value = self.mcts(
                    latent_state=latent_state,
                    critic_model=self.pred_net,
                    dynamics_model=self.dyn_net,
                )

                # TODO: Sample action from tree policy
                action = tree_policy[0]

                next_state, next_reward, done = self.env.step(action)
                chunk = Chunk(
                    state=state,
                    policy=tree_policy,
                    reward=next_reward,
                    value=value,
                    best_action=action,
                )
                epidata.chunks.append(chunk)
                state = next_state

                # TODO: Backpropagate actual reward found in terminal state

                if done:
                    break

            episode_history.append(epidata)

        return episode_history
