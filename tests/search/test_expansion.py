from src.search.expansion import expand_node, _transform_latent_state
from src.neural_network import DynamicsNetwork
from src.search.nodes import Node
import torch
import pytest


@pytest.mark.parametrize(
    ("actions", "latent_dim"),
    [
        (torch.tensor([0, 1, 2, 3]), 10),
        (torch.tensor([0, 1, 2, 3]), 5),
        (torch.tensor([0]), 5),
    ],
)
def test_expanding_leaf_node(actions, latent_dim):
    batch_size = 1
    possible_actions = actions.size(0)
    latent_state = torch.randn(batch_size, latent_dim)
    root = Node(latent_state)
    dynamics_network = DynamicsNetwork(latent_dim, possible_actions)
    assert len(root.children) == 0

    expand_node(root, actions, dynamics_network)
    assert len(root.children) == len(actions)


def test_latent_state_shape_transformer():
    batch_size = 1
    latent_state = torch.Tensor([[1, 2, 3]])
    expected = torch.Tensor([[1, 2, 3] * batch_size])
    actual = _transform_latent_state(batch_size, latent_state)
    assert actual.shape == expected.shape
