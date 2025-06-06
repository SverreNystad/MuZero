from dataclasses import dataclass, field
from typing import Tuple, Union

import torch


@dataclass
class Node:
    """
    Represents a node in the MCTS search tree. """

    latent_state: torch.Tensor
    parent: Union["Node", None] = None
    children: dict[torch.Tensor, "Node"] = field(default_factory=dict)
    to_play: int = field(default_factory=int)
    visit_count: int = field(default_factory=int)
    value_sum: float = field(default_factory=float)
    reward: float = field(default_factory=float)
    policy_priority: float = field(default_factory=float)

    # for visualization purposes
    pos: Tuple[int, int] = field(default_factory=lambda: (400, 100))

    def add_child(self, latent_state: torch.Tensor, action: torch.Tensor) -> "Node":
        """
        Adds a child to the node.
        """
        child = Node(latent_state, parent=self, to_play=-self.to_play)
        self.children[action] = child
        return child
