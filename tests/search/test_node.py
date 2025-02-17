import torch
from src.search.nodes import Node


def test_node_creation():
    root = Node(torch.tensor([0, 0, 0, 0]), to_play=1)
    assert root.latent_state.tolist() == [0, 0, 0, 0]
    assert root.to_play == 1
    assert root.visit_count == 0
    assert root.value_sum == 0.0
    assert root.reward == 0.0


def test_add_several_children():
    root = Node(torch.tensor([0, 0, 0, 0]), to_play=1)
    node_1 = Node(torch.tensor([1, 2, 3, 4]), to_play=-1)
    node_2 = Node(torch.tensor([2, 3, 4, 5]), to_play=-1)

    action_1 = torch.tensor([0, 0])
    action_2 = torch.tensor([0, 1])

    root.add_child(node_1.latent_state, action_1)
    root.add_child(node_2.latent_state, action_2)

    assert len(root.children) == 2
    assert action_1 in root.children.keys()
    assert action_2 in root.children.keys()


def test_add_same_child_twice():
    root = Node(torch.tensor([0, 0, 0, 0]), to_play=1)
    node_1 = Node(torch.tensor([1, 2, 3, 4]), to_play=-1)

    action_1 = torch.tensor([0, 0])

    root.add_child(node_1.latent_state, action_1)
    root.add_child(node_1.latent_state, action_1)

    assert len(root.children) == 1
    assert action_1 in root.children.keys()
