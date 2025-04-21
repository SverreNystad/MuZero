import ray
import torch

import wandb
from src.config.config_loader import TrainingConfig
from src.neural_networks.neural_network import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)


@ray.remote
class ParameterServer:
    def __init__(
        self,
        representation_network: RepresentationNetwork,
        dynamics_networks: DynamicsNetwork,
        prediction_network: PredictionNetwork,
        config: TrainingConfig,
    ):
        self.representation_network = representation_network
        self.dynamics_networks = dynamics_networks
        self.prediction_network = prediction_network
        self.optimizer = torch.optim.Adam(
            list(self.representation_network.parameters())
            + list(self.dynamics_networks.parameters())
            + list(self.prediction_network.parameters()),
            lr=config.learning_rate,
            betas=config.betas,
        )
        wandb.init()
        wandb.watch(representation_network, log="all")
        wandb.watch(dynamics_networks, log="all")
        wandb.watch(prediction_network, log="all")

    def get_networks(self) -> tuple[RepresentationNetwork, DynamicsNetwork, PredictionNetwork]:
        return (
            self.representation_network,
            self.dynamics_networks,
            self.prediction_network,
        )

    def apply_gradients(self, grads):
        self.optimizer.zero_grad()
        for param, grad in zip(
            list(self.representation_network.parameters())
            + list(self.dynamics_networks.parameters())
            + list(self.prediction_network.parameters()),
            grads,
        ):
            param.grad = grad
        self.optimizer.step()
