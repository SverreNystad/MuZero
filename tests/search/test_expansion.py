from src.search.expansion import expand_node
from src.neural_network import DynamicsNetwork
from src.search.nodes import Node
import torch
import pytest


@pytest.mark.parametrize(
    ("batch_size", "action_space", "latent_dim"),
    [
        (2, 2, 10),
        (3, 3, 5),
    ],
)
def test_expanding_leaf_node(batch_size, action_space, latent_dim):
    latent_state = torch.randn(batch_size, latent_dim)
    root = Node(latent_state)
    dynamics_network = DynamicsNetwork(latent_dim, action_space)
    assert len(root.children) == 0
    possible_actions = []
    for i in range(action_space):
        possible_actions.append(torch.tensor([i] * batch_size))

    expand_node(root, possible_actions, dynamics_network)
    assert len(root.children) == len(possible_actions)
