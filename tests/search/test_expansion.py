import pytest
import torch

from src.search.expansion import _transform_latent_state, expand_node
from src.search.nodes import Node
from tests.nerual_networks.test_networks import tiny_dyn_net, tiny_pred_net


@pytest.mark.parametrize(
    ("actions", "latent_shape"),
    [
        (torch.tensor([0, 1, 2, 3]), (10, 2, 6)),
        (torch.tensor([0, 1, 2, 3]), (5, 2, 6)),
        (torch.tensor([0]), (5, 2, 6)),
    ],
)
def test_expanding_leaf_node(actions, latent_shape):
    batch_size = 1
    possible_actions = actions.size(0)
    latent_state = torch.randn(batch_size, *latent_shape)
    root = Node(latent_state)
    dynamics_network = tiny_dyn_net(latent_shape, num_actions=possible_actions)
    prediction_network = tiny_pred_net(latent_shape, num_actions=possible_actions)
    assert len(root.children) == 0

    expand_node(root, actions, dynamics_network, prediction_network)
    assert len(root.children) == len(actions)
    # Check that the latent state of the root node is not changed
    assert torch.allclose(root.latent_state, latent_state)

    # Check that the latent state of the children has correct shape
    for child in root.children.values():
        assert child.latent_state.shape == latent_state.shape


@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
)
def test_latent_state_shape_transformer(batch_size):
    latent_state = torch.Tensor([[1, 2, 3]])
    # Expected shape: (batch_size, C, H, W) => (1, 1, 1, 1)
    expected = torch.Tensor([[[[1, 2, 3]]]] * batch_size)
    actual = _transform_latent_state(batch_size, latent_state)
    assert actual.shape == expected.shape
