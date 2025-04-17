import ray
import torch
from torch._prims_common import DeviceLikeType

from src.config.config_loader import Configuration
from src.environments.factory import create_environment
from src.neural_networks.neural_network import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from src.parameter_server import ParameterServer
from src.replay_buffer import ReplayBuffer
from src.training_data_generator import (
    Episode,
    TrainingDataGenerator,
    save_training_data,
)


@ray.remote
class Actor:
    """
    Actor class for continuos self play
    """

    def __init__(self, replay_buffer: ReplayBuffer, parameter_server: ParameterServer):
        self.replay_buffer = replay_buffer
        self.parameter_server = parameter_server

    def start_continuos_self_play(
        self,
        config: Configuration,
        device: DeviceLikeType = "cpu",  # Maybe check if computer has GPU and use it
    ) -> None:
        """
        Start the continuous self-play process.
        """

        while True:
            # Load the neural networks from the parameter server.
            repr_net, dyn_net, pred_net = ray.get(self.parameter_server.get_networks.remote())
            training_steps = ray.get(self.parameter_server.training_steps.remote())

            # Generate training data.
            episodes = generate_training_data(
                repr_net=repr_net,
                dyn_net=dyn_net,
                pred_net=pred_net,
                config=config,
                device=device,
                training_steps=training_steps,
                save_data=False,
            )
            # Add the generated episodes to the replay buffer.
            self.replay_buffer.add_episodes.remote(episodes)

            # TODO: ADD POSSIBILITY TO STOP SELF PLAY


@torch.no_grad()
def generate_training_data(
    repr_net: RepresentationNetwork,
    dyn_net: DynamicsNetwork,
    pred_net: PredictionNetwork,
    config: Configuration,
    device: DeviceLikeType = "cpu",
    training_steps: int = 0,
    save_data: bool = False,
) -> list[Episode]:
    """
    Generate training data for the training loop.
    """
    # Create the environment using the factory method.
    env = create_environment(config.environment, device)
    # Create the training data generator.
    training_data_generator = TrainingDataGenerator(
        env=env,
        repr_net=repr_net,
        dyn_net=dyn_net,
        pred_net=pred_net,
        config=config.training_data_generator,
        device=device,
    )
    # Generate the training data.
    episodes = training_data_generator.generate_training_data(training_steps)

    if save_data:
        path = save_training_data(episodes)
        print(f"Training data saved to {path}")
    return episodes
