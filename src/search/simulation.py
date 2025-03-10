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
        latent_state_batched = node.latent_state.unsqueeze(0)  # (1, latent_dim)

        value: torch.Tensor
        _, value = self.predictor(latent_state_batched)

        return value.item()
