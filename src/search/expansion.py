import random

import torch

from src.neural_networks.neural_network import DynamicsNetwork, PredictionNetwork
from src.search.nodes import Node


def expand_node(
    node: Node,
    actions: torch.Tensor,
    dynamics_network: DynamicsNetwork,
    prediction_network: PredictionNetwork,
) -> Node:
    """
    Expand the given node by adding all child nodes (based on actions) and return one of them

    Args:
        - node (Node): The node to expand from
    """
    batch_size = actions.size(0)
    if batch_size <= 0:
        raise ValueError("No possible actions to expand the node with")

    transformed_latent_state = _transform_latent_state(batch_size, node.latent_state)
    policy, _ = prediction_network(node.latent_state)
    policy = torch.softmax(policy, dim=1).squeeze(0)

    next_latent_states, rewards = dynamics_network(transformed_latent_state, actions)
    for i in range(batch_size):
        child = node.add_child(next_latent_states[i].unsqueeze(0), actions[i]) # (4, 6, 6, 3) -> [4 * [...]], (6, 6, 3) -> unsqueeze -> (1, 6, 6, 3)
        child.policy_priority = policy[i].item() #shape (1, 4) -> [[1, 1, 1, 1]]
        child.reward = rewards[i]

    return random.choice(list(node.children.values()))


def _transform_latent_state(batch_size: int, latent_state: torch.Tensor) -> torch.Tensor:
    """Transforms the latent state into the correct shape for the batch size"""
    return latent_state.repeat(batch_size, 1, 1, 1)
