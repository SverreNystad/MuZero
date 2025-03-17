"""
This module contains the interfaces for the different strategies used in the MCTS algorithm.
To read more about the strategy design pattern, see https://refactoring.guru/design-patterns/strategy
"""

from typing import Protocol
from src.search.nodes import Node


class SelectionStrategy(Protocol):
    def __call__(self, root: Node) -> Node:
        """
        Select a node from the tree starting at `root` for further expansion.

        Args:
            - root (Node) The root node of the tree.
        Returns:
            The selected node.
        """
        ...


class SimulationStrategy(Protocol):
    def __call__(self, node: Node) -> list[float]:
        """
        Simulate a play-out (rollout) from the given node to the end of the game.

        Args:
            - node (Node) The node from which to start the simulation.
        Returns:
            The result of the simulation
        """
        ...


class BackpropagationStrategy(Protocol):
    def __call__(self, node: Node, rewards: list[float], to_play: int) -> None:
        """
        Backpropagate the simulation result up through the tree.

        Args:
            - node (Node) The node from which the simulation started.
            - result (float) The result of the simulation
        """
        # TODO: Find out how to determine the flipping of the result for adversarial games and non-adversarial games and for team games
        ...
