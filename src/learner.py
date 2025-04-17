import ray
from torch._prims_common import DeviceLikeType

from src.config.config_loader import TrainingConfig
from src.parameter_server import ParameterServer
from src.replay_buffer import ReplayBuffer


@ray.remote
class Learner:
    """
    Learner class for training the neural networks.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        parameter_server: ParameterServer,
    ):
        """
        Args:
            replay_buffer(ReplayBuffer): Replay buffer for storing episodes
            parameter_server(ParameterServer): Parameter server for managing neural networks
        """

        self.replay_buffer = replay_buffer
        self.parameter_server = parameter_server

    def continuous_training(
        self,
        config: TrainingConfig,
        device: DeviceLikeType = "cpu",
    ) -> None:
        """
        Start the continuous training process.
        """
        while True:
            # Sample a batch of episodes from the replay buffer.
            batch, indices = ray.get(self.replay_buffer.sample_batch.remote())

            # Load the neural networks from the parameter server.
            repr_net, dyn_net, pred_net = ray.get(self.parameter_server.get_networks.remote())

            # Train the neural networks using the sampled batch.
            self.train_networks(repr_net, dyn_net, pred_net, batch)

            # Update the priorities of the sampled episodes in the replay buffer.
            self.replay_buffer.update_priorities.remote(indices)
            # Update the neural networks in the parameter server.
