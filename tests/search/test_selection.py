import pytest
import torch

from src.search.nodes import Node
from src.search.selection import PUCT, UCT
from src.search.strategies import SelectionStrategy


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
    latent_state = torch.randn(1, 3, 3, 3)
    root = Node(latent_state)
    root.visit_count = 16
    child1 = root.add_child(latent_state, torch.tensor(1))
    child1.visit_count = 10
    child1.value_sum = 1
    child2 = root.add_child(latent_state, torch.tensor(2))
    child2.visit_count = 5
    child2.value_sum = 10
    assert strategy(root) == child2
