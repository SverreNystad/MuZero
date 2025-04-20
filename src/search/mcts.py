import math
import time

import torch

from src.neural_networks.neural_network import DynamicsNetwork, PredictionNetwork
from src.search.expansion import expand_node
from src.search.nodes import Node
from src.search.strategies import (
    BackpropagationStrategy,
    SelectionStrategy,
    SimulationStrategy,
)


class MCTS:
    def __init__(
        self,
        selection: SelectionStrategy,
        simulation: SimulationStrategy,
        backpropagation: BackpropagationStrategy,
        dynamic_network: DynamicsNetwork,
        prediction_network: PredictionNetwork,
        actions: torch.Tensor,
        max_itr: int = 0,
        max_time: float = 0.0,
        dirichlet_alpha: float = 0.3,
        noise_frac: float = 0.25,
    ) -> None:
        self.selection = selection
        self.simulation = simulation
        self.backpropagation = backpropagation
        self.actions = actions
        self.dynamics_network = dynamic_network
        self.prediction_network = prediction_network
        self.max_itr = max_itr
        self.max_time = max_time
        self.dirichlet_alpha: float = dirichlet_alpha
        self.noise_frac: float = noise_frac

    def run(self, root: Node) -> tuple[list[float], float]:
        """
        Run the Monte Carlo Tree Search algorithm mutating the tree starting at `root`.
        """

        # Add Dirichlet noise to the root node.
        self._add_dirichlet_noise(root)

        if self.max_itr == 0:
            start_time = time.time()
            while time.time() - start_time < self.max_time:
                self._step(root)
        else:
            itr = 0
            while itr < self.max_itr:
                self._step(root)
                itr += 1

        utility = root.value_sum / root.visit_count
        tree_policy = _soft_max([child_node.value_sum for child_node in root.children.values()])

        return tree_policy, utility

    def _step(self, node: Node) -> None:
        """
        Run a single step of the Monte Carlo Tree Search algorithm.
        """
        chosen_node = self.selection(node)
        expanded_node = expand_node(chosen_node, self.actions, self.dynamics_network, self.prediction_network)
        rewards = self.simulation(expanded_node)
        self.backpropagation(expanded_node, rewards, chosen_node.to_play)

    def _add_dirichlet_noise(self, root: Node) -> None:
        """
        Add Dirichlet noise to the node.
        """
        expand_node(root, self.actions, self.dynamics_network, self.prediction_network)
        num_actions = len(root.children)
        noise = torch.distributions.Dirichlet(torch.full((num_actions,), self.dirichlet_alpha)).sample()
        for child, dirichlet_sample in zip(root.children.values(), noise):
            # new_prior = (1‑ε)·P + ε·η
            child.policy_priority += (1.0 - self.noise_frac) * child.policy_priority + self.noise_frac * dirichlet_sample


def _soft_max(values: list[float]) -> list[float]:
    """
    Compute the softmax of vector x in a numerically stable way.
    """
    exp_values = [math.exp(x) for x in values]
    sum_exp = sum(exp_values)
    return [x / sum_exp for x in exp_values]
