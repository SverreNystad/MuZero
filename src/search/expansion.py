from src.neural_network import DynamicsNetwork
from src.search.nodes import Node
from torch import Tensor, randn
import random


def expand_node(
    node: Node, possible_actions: list[Tensor], dynamics_network: DynamicsNetwork
) -> Node:
    """
    Expand the given node by adding one or more child nodes and return one of them

    Args:
        - node (Node): The node to expand from
    """
    for action in possible_actions:
        next_latent_state, reward = dynamics_network(node.latent_state, action)
        child = node.add_child(next_latent_state, action)
        child.reward = reward
    return random.choice(list(node.children.values()))
