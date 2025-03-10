from src.search.strategies import BackpropagationStrategy
from src.search.nodes import Node
from src.search.backpropagation import Backpropagation
import pytest
import torch


@pytest.mark.parametrize(
    "strategy",
    [
        (Backpropagation()),
    ],
)
def test_backpropagation_single_player(strategy: BackpropagationStrategy):
    latent_state = torch.randn(1, 3, 3)
    player = 1
    root = Node(latent_state)
    root.visit_count = 16
    root.value_sum = 20
    root.to_play = player
    child1 = root.add_child(latent_state, torch.tensor(1))
    child1.visit_count = 10
    child1.to_play = player
    child1.value_sum = 5
    child2 = root.add_child(latent_state, torch.tensor(2))
    child2.visit_count = 5
    child2.to_play = player
    child2.value_sum = 10

    # Act
    strategy(child2, 5, player)

    # Assert
    # The backpropagation should update the value_sum and visit_count of the child and parent
    assert child2.visit_count == 6
    assert child2.value_sum == 15
    assert root.visit_count == 17
    assert root.value_sum == 25

    # The backpropagation should not update the value_sum and visit_count of the other child
    assert child1.visit_count == 10
    assert child1.value_sum == 5


@pytest.mark.parametrize(
    "strategy",
    [
        (Backpropagation()),
    ],
)
def test_backpropagation_two_player(strategy: BackpropagationStrategy):
    latent_state = torch.randn(1, 3, 3)
    player_1 = 1
    player_2 = 2
    root = Node(latent_state)
    root.visit_count = 16
    root.value_sum = 20
    root.to_play = player_1
    child1 = root.add_child(latent_state, torch.tensor(1))
    child1.visit_count = 10
    child1.to_play = player_2
    child1.value_sum = 5
    child2 = root.add_child(latent_state, torch.tensor(2))
    child2.visit_count = 5
    child2.to_play = player_2
    child2.value_sum = 10

    # Act
    strategy(child2, 5, player_2)

    # Assert
    # The backpropagation should update the value_sum and visit_count of the child and parent
    assert child2.visit_count == 6
    assert child2.value_sum == 15
    assert root.visit_count == 17
    assert root.value_sum == 15

    # The backpropagation should not update the value_sum and visit_count of the other child
    assert child1.visit_count == 10
    assert child1.value_sum == 5
