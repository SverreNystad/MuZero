import os
import pickle
import time
from tqdm import trange
from torch import Tensor
from dataclasses import dataclass

from src.environment import Environment
from src.neural_network import RepresentationNetwork, PredictionNetwork, DynamicsNetwork
from src.search.factory import create_mcts
from src.search.mcts import MCTS
from src.search.nodes import Node


@dataclass
class Chunk:
    """
    Data class to store chunk state data.
    """

    state: Tensor
    """ The raw observation (before representation network)"""
    policy: Tensor
    """ The policy distribution returned by MCTS (list -> converted to Tensor)"""
    reward: float
    """ The reward from the environment """
    value: float
    """ The value estimate from MCTS """
    best_action: int


@dataclass
class Episode:
    """
    Data class to store the entire sequence of `Chunk`s for an episode.
    """

    chunks: list[Chunk]


class TrainingDataGenerator:
    def __init__(
        self,
        env: Environment,
        repr_net: RepresentationNetwork,
        dyn_net: DynamicsNetwork,
        pred_net: PredictionNetwork,
        config: dict,
    ):
        """
        Args:
            env (Environment): Your environment.
            repr_net (RepresentationNetwork): Representation network for converting obs -> latent.
            dyn_net (DynamicsNetwork): Dynamics network for MCTS expansions.
            pred_net (PredictionNetwork): Prediction network for MCTS value/policy.
            config (dict): Must contain at least:
                - num_episodes: int
                - max_steps: int (or some max steps per episode)
                - max_time_mcts: float (seconds for MCTS if using time-based termination)
                - look_back: int (unused in this example, you can incorporate if needed)
        """
        self.env = env
        self.repr_net = repr_net
        self.dyn_net = dyn_net
        self.pred_net = pred_net

        self.num_episodes: int = config["num_episodes"]  # N_e
        self.max_steps: int = config["max_steps"]  # N_es
        self.look_back: int = config["look_back"]  # q
        self.total_time: float = config["total_time"]  # Not used below, but available
        self.mcts: MCTS = create_mcts(
            dynamics_network=dyn_net,
            prediction_network=pred_net,
            actions=self._make_actions_tensor(env),
            selection_type="puct",
            max_time=config["max_time_mcts"],  # 0 => iteration-based if you prefer
        )

    def _make_actions_tensor(self, env: Environment) -> Tensor:
        """
        Convert the environment's action space (e.g. range of valid moves) into a Torch tensor.
        """
        # E.g., if env.get_action_space() returns an integer (num_actions),
        # we can create a range tensor of that size.
        action_space = env.get_action_space()
        if isinstance(action_space, int):
            return Tensor(range(action_space)).long()
        # If your environment's action space is already a tuple or list, etc.
        return Tensor(action_space).long()

    def generate_training_data(self) -> list[Episode]:
        """
        Collect data by playing episodes with MCTS + neural networks.
        Returns:
            A list of `Episode` objects, each containing multiple `Chunk`s.
        """
        episode_history: list[Episode] = []

        for _ in trange(self.num_episodes, desc="Generating Episodes"):
            self.env.reset()
            # If your environment is multi-player, you might also reset `player_id` or handle it differently.

            epidata = Episode(chunks=[])
            state = self.env.get_state()  # Shape depends on your environment.

            for _ in trange(self.max_steps, desc="Steps per Episode", leave=False):
                # Convert the environment's state to a latent representation
                # or keep it as-is if the environment already gives a latent state.
                state = self.env.get_state()  # shape (3, 96, 96)
                state = state.unsqueeze(0)  # shape (1, 3, 96, 96)
                latent_state = self.repr_net(
                    state
                )  # Good! This matches the [N, C, H, W] convention.

                # Create an MCTS root node. If your environment is single-player, set `to_play=0`.
                root = Node(
                    latent_state=latent_state,
                    to_play=0,  # or the correct current player ID
                )

                # Run MCTS to get policy distribution (tree_policy) and value estimate.
                tree_policy, value = self.mcts.run(root)

                # Convert tree_policy to a torch.Tensor so we can store it in our Chunk easily.
                policy_tensor = Tensor(tree_policy)

                # Choose the best action by maximum probability in `tree_policy`.
                best_action = int(policy_tensor.argmax().item())

                # Step the environment
                next_state, next_reward, done = self.env.step(best_action)

                # Create a Chunk to store this transition
                chunk = Chunk(
                    state=state,
                    policy=policy_tensor,
                    reward=next_reward,
                    value=value,
                    best_action=best_action,
                )
                epidata.chunks.append(chunk)

                state = next_state

                if done:
                    break

            episode_history.append(epidata)

        return episode_history


DATA_FOLDER = "data/"


def save_training_data(training_data: list[Episode]) -> str:
    """
    Save the list of Episode objects as a binary file using pickle and return the path.
    """
    # Create the folder if it does not exist
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    date = time.time()
    filename = f"{DATA_FOLDER}training_episodes_{len(training_data)}_at_{date}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(training_data, f)
    return filename


def load_training_data(path: str) -> list[Episode]:
    """
    Load and return the list of Episode objects from the given path.
    """

    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_training_data() -> list[Episode]:
    all_files = os.listdir(DATA_FOLDER)
    all_episodes = []

    for file in all_files:
        file_path = os.path.join(DATA_FOLDER, file)

        # Check if the file is empty
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {file_path}")
            continue

        with open(file_path, "rb") as f:
            try:
                episodes = pickle.load(f)
                all_episodes.extend(episodes)
            except EOFError:
                print(f"Error loading {file_path}: file is corrupted or incomplete.")
                continue

    return all_episodes
