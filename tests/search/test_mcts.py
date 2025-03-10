import time
import torch
from torch import from_numpy, Tensor
from pydantic import BaseModel

from src.environments.connect_four import ConnectFour, ConnectFourConfig
from src.environment import Environment
from src.neural_network import (
    RepresentationNetwork,
    DynamicsNetwork,
    PredictionNetwork,
)
from src.search.factory import create_mcts
from src.search.nodes import Node


def preprocess_observation(
    observation: Tensor, obs_space: tuple
) -> tuple[Tensor, int, tuple]:
    """
    Preprocess the observation for the RepresentationNetwork.
    - Converts the observation to float.
    - If the observation is 2D (height, width), adds a channel dimension.
    - If the observation is (height, width, channels), permutes to (channels, height, width).
    In both cases, adds a batch dimension.
    """
    # Convert observation to float
    observation = observation.float()

    if len(obs_space) == 2:
        # Assume grayscale image.
        input_channels = 1
        height, width = obs_space
        if observation.dim() == 2:
            observation = observation.unsqueeze(0)
        observation = observation.unsqueeze(0)  # (1, 1, height, width)
    elif len(obs_space) == 3:
        # Assume observation is (height, width, channels)
        height, width, ch = obs_space
        input_channels = ch
        if observation.dim() == 3 and observation.shape[-1] == input_channels:
            observation = observation.permute(2, 0, 1)
        observation = observation.unsqueeze(0)  # (1, channels, height, width)
    else:
        raise ValueError("Unexpected observation shape")

    return observation, input_channels, (height, width)


def get_num_actions(env: Environment) -> int:
    """
    Determine the number of actions from the environment's action space.
    """
    action_space = env.get_action_space()  # Typically a tuple, e.g., (7,)
    if isinstance(action_space, tuple) and len(action_space) > 0:
        return action_space[0]
    return 7  # Fallback default


def test_mcts():
    """
    General test: Use the Connect Four environment and real networks to run MCTS
    and verify that the root node is updated.
    """
    # Create the environment using ConnectFour with a configuration.
    config = ConnectFourConfig(render_mode="rgb_array", seed=42)
    env = ConnectFour(config)

    # Get the initial observation from the environment.
    observation = env.reset()
    obs_space = env.get_observation_space()  # e.g., (6, 7) or (6, 7, channels)

    # Preprocess observation for the RepresentationNetwork.
    observation, input_channels, spatial_dims = preprocess_observation(
        observation, obs_space
    )
    height, width = spatial_dims

    # Determine the number of actions.
    num_actions = get_num_actions(env)

    # Instantiate the real networks.
    latent_dim = 16
    rep_net = RepresentationNetwork(
        input_channels=input_channels,
        observation_space=(height, width),
        latent_dim=latent_dim,
    )
    dyn_net = DynamicsNetwork(latent_dim=latent_dim, num_actions=num_actions)
    pred_net = PredictionNetwork(latent_dim=latent_dim, num_actions=num_actions)

    # Compute the latent state from the preprocessed observation.
    latent_state = rep_net(observation)

    # Define the set of possible actions.
    actions = torch.arange(num_actions)

    # Create the MCTS instance using the factory method (using 5 iterations here).
    mcts = create_mcts(
        dynamics_network=dyn_net,
        prediction_network=pred_net,
        actions=actions,
        selection_type="uct",
        max_itr=5,
        max_time=0.0,  # Iteration-based termination
    )

    # Create the root node using the computed latent state.
    root = Node(latent_state=latent_state, to_play=1, visit_count=0, value_sum=0.0)

    # Run MCTS.
    tree_policy, utility = mcts.run(root)

    # Check that the root node was updated.
    assert (
        root.visit_count > 0
    ), "Root node's visit count should be updated after MCTS run."
    print("test_mcts passed.")


def test_mcts_with_max_iterations():
    """
    Test: Ensure that MCTS terminates after the specified number of iterations.
    """
    config = ConnectFourConfig(render_mode="rgb_array", seed=42)
    env = ConnectFour(config)
    observation = env.reset()
    obs_space = env.get_observation_space()
    observation, input_channels, spatial_dims = preprocess_observation(
        observation, obs_space
    )
    height, width = spatial_dims
    num_actions = get_num_actions(env)

    latent_dim = 16
    rep_net = RepresentationNetwork(
        input_channels=input_channels,
        observation_space=(height, width),
        latent_dim=latent_dim,
    )
    dyn_net = DynamicsNetwork(latent_dim=latent_dim, num_actions=num_actions)
    pred_net = PredictionNetwork(latent_dim=latent_dim, num_actions=num_actions)

    latent_state = rep_net(observation)
    actions = torch.arange(num_actions)

    max_itr = 10
    mcts = create_mcts(
        dynamics_network=dyn_net,
        prediction_network=pred_net,
        actions=actions,
        selection_type="uct",
        max_itr=max_itr,
        max_time=0.0,  # Iteration-based termination.
    )

    root = Node(latent_state=latent_state, to_play=1, visit_count=0, value_sum=0.0)

    mcts.run(root)

    # Check that the root's visit_count equals the number of iterations.
    assert (
        root.visit_count == max_itr
    ), f"Expected {max_itr} iterations, got {root.visit_count}"
    print("test_mcts_with_max_iterations passed.")


def test_mcts_with_max_time():
    """
    Test: Ensure that MCTS stops running after approximately the specified max_time.
    """
    config = ConnectFourConfig(render_mode="rgb_array", seed=42)
    env = ConnectFour(config)
    observation = env.reset()
    obs_space = env.get_observation_space()
    observation, input_channels, spatial_dims = preprocess_observation(
        observation, obs_space
    )
    height, width = spatial_dims
    num_actions = get_num_actions(env)

    latent_dim = 16
    rep_net = RepresentationNetwork(
        input_channels=input_channels,
        observation_space=(height, width),
        latent_dim=latent_dim,
    )
    dyn_net = DynamicsNetwork(latent_dim=latent_dim, num_actions=num_actions)
    pred_net = PredictionNetwork(latent_dim=latent_dim, num_actions=num_actions)

    latent_state = rep_net(observation)
    actions = torch.arange(num_actions)

    max_time = 0.5  # seconds
    mcts = create_mcts(
        dynamics_network=dyn_net,
        prediction_network=pred_net,
        actions=actions,
        selection_type="uct",
        max_itr=0,  # Time-based termination mode.
        max_time=max_time,
    )

    root = Node(latent_state=latent_state, to_play=1, visit_count=0, value_sum=0.0)

    start_time = time.time()
    mcts.run(root)
    elapsed = time.time() - start_time

    # Verify that the elapsed time is at least max_time.
    assert (
        elapsed >= max_time
    ), f"Expected run to take at least {max_time} seconds, but took {elapsed:.2f} seconds"
    # Also ensure that some iterations occurred.
    assert (
        root.visit_count > 0
    ), "MCTS run did not update the root's visit count in time-based mode."
    print("test_mcts_with_max_time passed.")


if __name__ == "__main__":
    test_mcts()
    test_mcts_with_max_iterations()
    test_mcts_with_max_time()
