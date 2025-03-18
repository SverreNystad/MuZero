import time
import pytest
import torch

from src.environments.connect_four import ConnectFourConfig
from src.environments.car_racing import CarRacingConfig
from src.environments.factory import create_environment
from src.nerual_networks.neural_network import (
    RepresentationNetwork,
    DynamicsNetwork,
    PredictionNetwork,
)
from src.search.factory import MCTSConfig, create_mcts
from src.search.nodes import Node
from tests.nerual_networks.test_networks import (
    tiny_dyn_net,
    tiny_pred_net,
    tiny_repr_net,
)


@pytest.mark.parametrize(
    "env_config",
    [
        CarRacingConfig(seed=42, render_mode="rgb_array"),  # (3, 96, 96)
        ConnectFourConfig(),  # (2, 6, 7)
    ],
)
def test_mcts(env_config):
    """
    General test: Use the Connect Four environment and real networks to run MCTS
    and verify that the root node is updated.
    """
    # Create the environment using ConnectFour with a configuration.
    env = create_environment(env_config)

    # Get the initial observation from the environment.
    observation = env.reset()

    # Determine the number of actions.
    num_actions = len(env.get_action_space())
    # Instantiate the real networks.
    latent_shape = (2, 2, 2)
    repr_net = tiny_repr_net(
        latent_shape=latent_shape, observation_space=env.get_observation_space()
    )
    dyn_net = tiny_dyn_net(latent_shape, num_actions=num_actions)
    pred_net = tiny_pred_net(latent_shape, num_actions=num_actions)

    # Compute the latent state from the preprocessed observation.
    latent_state = repr_net(observation)

    # Define the set of possible actions.
    actions = torch.arange(num_actions)

    # Create the MCTS instance using the factory method (using 5 iterations here).
    config = MCTSConfig(
        selection_strategy="uct",
        max_iterations=5,
        max_time=0.0,  # Iteration-based termination
        depth=1,
        discount_factor=1.0,
    )
    mcts = create_mcts(
        dynamics_network=dyn_net,
        prediction_network=pred_net,
        actions=actions,
        config=config,
    )

    # Create the root node using the computed latent state.
    root = Node(latent_state=latent_state, to_play=1, visit_count=0, value_sum=0.0)

    # Run MCTS.
    tree_policy, utility = mcts.run(root)

    # Check that the root node was updated.
    assert (
        root.visit_count > 0
    ), "Root node's visit count should be updated after MCTS run."

    assert utility is not None, "Utility should not be None after MCTS run."
    assert tree_policy is not None, "Tree policy should not be None after MCTS run."
    assert isinstance(utility, float), "Utility should be a float."
    assert (
        isinstance(tree_policy, list) and len(tree_policy) == num_actions
    ), "Tree policy should be a list of length equal to the number of actions."


@pytest.mark.parametrize(
    "env_config",
    [
        CarRacingConfig(seed=42, render_mode="rgb_array"),  # (3, 96, 96)
        ConnectFourConfig(),  # (2, 6, 7)
    ],
)
def test_mcts_with_max_iterations(env_config):
    """
    Test: Ensure that MCTS terminates after the specified number of iterations.
    """
    # Create the environment using ConnectFour with a configuration.
    env = create_environment(env_config)

    # Get the initial observation from the environment.
    observation = env.reset()

    # Determine the number of actions.
    num_actions = len(env.get_action_space())
    # Instantiate the real networks.
    latent_shape = (2, 2, 2)
    repr_net = tiny_repr_net(
        latent_shape=latent_shape, observation_space=env.get_observation_space()
    )
    dyn_net = tiny_dyn_net(latent_shape, num_actions=num_actions)
    pred_net = tiny_pred_net(latent_shape, num_actions=num_actions)

    latent_state = repr_net(observation)
    actions = torch.arange(num_actions)

    max_itr = 10
    config = MCTSConfig(
        selection_strategy="uct",
        max_iterations=max_itr,
        max_time=0.0,  # Iteration-based termination.
        discount_factor=1.0,
        depth=1,
    )
    mcts = create_mcts(
        dynamics_network=dyn_net,
        prediction_network=pred_net,
        actions=actions,
        config=config,
    )

    root = Node(latent_state=latent_state, to_play=1, visit_count=0, value_sum=0.0)

    tree_policy, utility = mcts.run(root)

    # Check that the root's visit_count equals the number of iterations.
    assert (
        root.visit_count == max_itr
    ), f"Expected {max_itr} iterations, got {root.visit_count}"

    assert utility is not None, "Utility should not be None after MCTS run."
    assert tree_policy is not None, "Tree policy should not be None after MCTS run."
    assert isinstance(utility, float), "Utility should be a float."
    assert (
        isinstance(tree_policy, list) and len(tree_policy) == num_actions
    ), "Tree policy should be a list of length equal to the number of actions."


@pytest.mark.parametrize(
    "env_config",
    [
        CarRacingConfig(seed=42, render_mode="rgb_array"),  # (3, 96, 96)
        ConnectFourConfig(),  # (2, 6, 7)
    ],
)
def test_mcts_with_max_time(env_config):
    """
    Test: Ensure that MCTS stops running after approximately the specified max_time.
    """
    # Create the environment using ConnectFour with a configuration.
    env = create_environment(env_config)

    # Determine the number of actions.
    num_actions = len(env.get_action_space())
    # Get the initial observation from the environment.
    observation = env.reset()
    latent_shape = (2, 2, 2)
    # Instantiate the real networks.
    repr_net = tiny_repr_net(
        latent_shape=latent_shape, observation_space=env.get_observation_space()
    )
    dyn_net = tiny_dyn_net(latent_shape, num_actions=num_actions)
    pred_net = tiny_pred_net(latent_shape, num_actions=num_actions)

    latent_state = repr_net(observation)
    actions = torch.arange(num_actions)

    max_time = 1  # seconds
    config = MCTSConfig(
        selection_strategy="uct",
        max_iterations=0,  # Time-based termination mode.
        max_time=max_time,
        depth=1,
    )
    mcts = create_mcts(
        dynamics_network=dyn_net,
        prediction_network=pred_net,
        actions=actions,
        config=config,
    )

    root = Node(latent_state=latent_state, to_play=1, visit_count=0, value_sum=0.0)

    start_time = time.time()
    tree_policy, utility = mcts.run(root)
    elapsed = time.time() - start_time

    # Verify that the elapsed time is at least max_time.
    assert (
        elapsed >= max_time
    ), f"Expected run to take at least {max_time} seconds, but took {elapsed:.2f} seconds"
    # Also ensure that some iterations occurred.
    assert (
        root.visit_count > 0
    ), "MCTS run did not update the root's visit count in time-based mode."
    assert utility is not None, "Utility should not be None after MCTS run."
    assert tree_policy is not None, "Tree policy should not be None after MCTS run."
    assert isinstance(utility, float), "Utility should be a float."
    assert (
        isinstance(tree_policy, list) and len(tree_policy) == num_actions
    ), "Tree policy should be a list of length equal to the number of actions."
