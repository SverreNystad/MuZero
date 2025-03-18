from src.nerual_networks.neural_network import DynamicsNetwork
from src.search.nodes import Node
import random, torch


def expand_node(
    node: Node,
    actions: torch.Tensor,
    dynamics_network: DynamicsNetwork,
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

    next_latent_states, rewards = dynamics_network(transformed_latent_state, actions)
    for i in range(batch_size):
        child = node.add_child(next_latent_states[i], actions[i])
        child.reward = rewards[i]

    return random.choice(list(node.children.values()))


def _transform_latent_state(
    batch_size: int, latent_state: torch.Tensor
) -> torch.Tensor:
    """Transforms the latent state into the correct shape for the batch size"""
    return latent_state.repeat(batch_size, 1, 1, 1)
