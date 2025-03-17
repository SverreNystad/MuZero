from typing import cast

from torch import Tensor, tensor
from src.nerual_networks.neural_network import DynamicsNetwork, PredictionNetwork
from src.search.nodes import Node
from src.search.strategies import SimulationStrategy


class MuZeroSimulation(SimulationStrategy):
    def __init__(
        self,
        dynamic_network: DynamicsNetwork,
        predictor: PredictionNetwork,
        depth: int,
    ) -> None:
        self.dynamic_network = dynamic_network
        self.predictor = predictor
        self.depth = depth

    def __call__(self, node: Node) -> list[float]:
        """
        Simulate a play-out (rollout) from the given node to the end of the game.

        Args:
            - node (Node) The node from which to start the simulation.
        Returns:
            The result of the simulation
        """
        latent_state_batched = node.latent_state.unsqueeze(0)  # (1, latent_dim)
        accumulated_reward = []
        for _ in range(self.depth):
            policy, value = self.predictor(latent_state_batched)
            action = tensor([policy.argmax().item()])  # shape [1]
            next_latent_state, reward = self.dynamic_network(
                latent_state_batched, action
            )
            accumulated_reward.append(reward)
            latent_state_batched = next_latent_state

        _, value = self.predictor(latent_state_batched)
        accumulated_reward.append(value)
        # Take out values from tensors
        accumulated_reward = [
            cast(float, reward.item()) for reward in accumulated_reward
        ]
        return accumulated_reward
