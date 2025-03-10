import torch
from src.neural_network import PredictionNetwork
from src.search.nodes import Node
from src.search.strategies import SimulationStrategy


class MuZeroSimulation(SimulationStrategy):
    def __init__(self, predictor: PredictionNetwork) -> None:
        self.predictor = predictor

    def __call__(self, node: Node) -> float:
        """
        Simulate a play-out (rollout) from the given node to the end of the game.

        Args:
            - node (Node) The node from which to start the simulation.
        Returns:
            The result of the simulation
        """
        value: torch.Tensor
        _, value = self.predictor(node.latent_state)

        return value.item()
