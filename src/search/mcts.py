import time
import torch

from src.neural_network import DynamicsNetwork
from src.search.nodes import Node
from src.search.strategies import (
    SelectionStrategy,
    SimulationStrategy,
    BackpropagationStrategy,
)
from src.search.expansion import expand_node


class MCTS:

    def __init__(
        self,
        selection: SelectionStrategy,
        simulation: SimulationStrategy,
        backpropagation: BackpropagationStrategy,
        dynamic_network: DynamicsNetwork,
        actions: torch.Tensor,
        max_itr: int = 0,
        max_time: float = 0.0,
    ) -> None:
        self.selection = selection
        self.simulation = simulation
        self.backpropagation = backpropagation
        self.actions = actions
        self.dynamics_network = dynamic_network
        self.max_itr = max_itr
        self.max_time = max_time

    def run(self, root: Node) -> None:
        """
        Run the Monte Carlo Tree Search algorithm mutating the tree starting at `root`.
        """
        if self.max_itr == 0:
            start_time = time.time()
            while time.time() - start_time < self.max_time:
                chosen_node = self.selection(root)
                created_node = expand_node(
                    chosen_node, self.actions, self.dynamics_network
                )
                result = self.simulation(created_node)
                self.backpropagation(created_node, result, root.to_play)
        else:
            itr = 0
            while itr < self.max_itr:
                _step(root)
                itr += 1

        # TODO: Calculate the action probabilities and the value of the root node

    def _step(self, node: Node) -> None:
        """
        Run a single step of the Monte Carlo Tree Search algorithm.
        """
        chosen_node = self.selection(node)
        created_node = expand_node(chosen_node, self.actions, self.dynamics_network)
        result = self.simulation(created_node)
        self.backpropagation(created_node, result, chosen_node.to_play)
