import os
import pickle
import time
from dataclasses import dataclass
from typing import cast

from torch import Tensor
from tqdm import trange

from src.config.config_loader import TrainingDataGeneratorConfig
from src.environment import Environment
from src.neural_networks.neural_network import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
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
        config: TrainingDataGeneratorConfig,
        device="cpu",
    ):
        """
        Args:
            env (Environment): Your environment.
            repr_net (RepresentationNetwork): Representation network for converting obs -> latent.
            dyn_net (DynamicsNetwork): Dynamics network for MCTS expansions.
            pred_net (PredictionNetwork): Prediction network for MCTS value/policy.
            config (TrainingDataGeneratorConfig): Configuration for the data generator.
            device (str): Device to run the networks on (e.g., "cpu" or "cuda").
        """
        self.env = env
        self.repr_net = repr_net
        self.dyn_net = dyn_net
        self.pred_net = pred_net
        self.device = device

        self.num_episodes: int = config.num_episodes  # N_e
        self.max_steps: int = config.max_steps_per_episode  # N_es
        self.mcts: MCTS = create_mcts(
            dynamics_network=dyn_net,
            prediction_network=pred_net,
            actions=self._make_actions_tensor(env),
            config=config.mcts,
            device=device,
        )

    def _make_actions_tensor(self, env: Environment) -> Tensor:
        """
        Convert the environment's action space (e.g. range of valid moves) into a Torch tensor.
        """
        # E.g., if env.get_action_space() returns an integer (num_actions),
        # we can create a range tensor of that size.
        action_space = env.get_action_space()
        if isinstance(action_space, int):
            return Tensor(range(action_space)).long().to(self.device)
        # If your environment's action space is already a tuple or list, etc.
        return Tensor(action_space).long().to(self.device)

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
                state = self.env.get_state()
                latent_state = self.repr_net(state)

                # Run MCTS to get policy distribution (tree_policy) and value estimate.
                root = Node(latent_state=latent_state, to_play=self.env.get_to_play())
                tree_policy, value = self.mcts.run(root)

                policy_tensor = Tensor(tree_policy).to(self.device)

                # Choose the best action by maximum probability in `tree_policy`.
                best_action = int(policy_tensor.argmax().item())

                next_state, next_reward, done = self.env.step(best_action)

                # Store the chunk data in the episode
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
        episodes = pickle.load(f)
    _validate_data(episodes)

    return cast(list[Episode], episodes)


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

    _validate_data(all_episodes)
    return cast(list[Episode], all_episodes)


def _validate_data(data: list[Episode]) -> None:
    """
    Validate the data by checking the type of each object in the list.
    """
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of Episode objects, but got {type(data)}")
    for episode in data:
        if not isinstance(episode, Episode):
            raise ValueError(f"Expected an Episode object, but got {type(episode)}")


def delete_all_training_data() -> None:
    """
    Delete all training data files in the data folder.
    """
    all_files = os.listdir(DATA_FOLDER)
    print(f"Deleting {len(all_files)} training data files...")
    for file in all_files:
        file_path = os.path.join(DATA_FOLDER, file)
        os.remove(file_path)
    print("All training data files deleted.")
