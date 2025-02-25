from src.search.strategies import SelectionStrategy
from src.search.nodes import Node
from src.search.selection import UCT, PUCT
import pytest
import torch


@pytest.mark.parametrize(
    "strategy",
    [
        (UCT()),
        (PUCT()),
    ],
)
def test_search_uct_when_root_is_empty_then_return_root(strategy: SelectionStrategy):
    latent_state = torch.randn(1, 3, 3)
    root = Node(latent_state)
    assert strategy(root) == root


@pytest.mark.parametrize(
    "strategy",
    [
        (UCT()),
        (PUCT()),
    ],
)
def test_search_uct_when_root_has_children_then_return_best_child(
    strategy: SelectionStrategy,
):
    latent_state = torch.randn(1, 3, 3)
    root = Node(latent_state)
    root.visit_count = 1
    child1 = root.add_child(latent_state, torch.tensor(1))
    child1.visit_count = 10
    child1.reward = 5
    child2 = root.add_child(latent_state, torch.tensor(2))
    child2.visit_count = 5
    child2.reward = 10
    assert strategy(root) == child2
