from dataclasses import dataclass
import time
import torch

from src.search.nodes import Node
from src.search.strategies import (
    SelectionStrategy,
    ExpansionStrategy,
    SimulationStrategy,
    BackpropagationStrategy,
)


@dataclass
class MCTSOutput:
    """
    Output of the MCTS algorithm.
    """

    action_probs: torch.Tensor
    value: torch.float


class MCTS:

    def __init__(
        self,
        selection: SelectionStrategy,
        expansion: ExpansionStrategy,
        simulation: SimulationStrategy,
        backpropagation: BackpropagationStrategy,
        max_itr: int = 0,
        max_time: float = 0.0,
    ) -> None:
        self.selection = selection
        self.expansion = expansion
        self.simulation = simulation
        self.backpropagation = backpropagation
        self.max_itr = max_itr
        self.max_time = max_time

    def run(self, root: Node) -> MCTSOutput:
        """
        Run the Monte Carlo Tree Search algorithm mutating the tree starting at `root`.
        """
        if self.max_itr == 0:
            start_time = time.time()
            while time.time() - start_time < self.max_time:
                chosen_node = self.selection(root)
                created_node = self.expansion(chosen_node)
                result = self.simulation(created_node)
                self.backpropagation(created_node, result)
        else:
            itr = 0
            while itr < self.max_itr:
                chosen_node = self.selection(root)
                created_node = self.expansion(chosen_node)
                result = self.simulation(created_node)
                self.backpropagation(created_node, result)
                itr += 1

        # TODO: Calculate the action probabilities and the value of the root node
