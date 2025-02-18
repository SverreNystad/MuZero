from tqdm import trange
from torch import Tensor
from dataclasses import dataclass
from src.environment import Environment
from src.neural_network import RepresentationNetwork, PredictionNetwork, DynamicsNetwork

@dataclass
class GameState:
    """
    Data class to store game state data.
    """
    state: Environment
    policy: Tensor
    reward: float
    value: float # TODO: Could be a Tensor with shape (1, 1)
    best_action: Tensor # TODO: Could be an int

@dataclass
class Episode:
    """
    Data class to store episode data.
    """
    states: list[GameState]

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
        self.num_episodes: int = config["num_episodes"] # N_e
        self.max_steps: int = config["max_steps"] # N_es
        self.look_back: int = config["look_back"] # q
        self.total_time: float = config["total_time"]

    def generate_training_data(self) -> list[Episode]:
        """
        Args:
            num_episodes: Number of episodes to run.
            total_time: Total time to run the episodes in milliseconds (ms).

        Returns:
            List of episode data. 
        """
        episode_history = []

        for _ in trange(self.num_episodes):
            self.env.reset()

            epidata = Episode(states=[])

            for _ in trange(self.max_steps):
                latent_state = self.repr_net(self.env.state)

                tree_policy: Tensor
                tree_policy, value = self.mcts(latent_state=latent_state, critic_model=self.pred_net, dynamics_model=self.dyn_net)

                # TODO: Sample action from tree policy
                action = tree_policy[0]
                
                self.env.step(action)
                epidata.states.append(self.env.state)

            episode_history.append(epidata)

            # TODO: Backpropagate actuale reward found in terminal state

        return episode_history