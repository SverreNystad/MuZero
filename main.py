import random

import numpy as np
import torch

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import os  # noqa: E402
from collections.abc import Callable  # noqa: E402

import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from torch._prims_common import DeviceLikeType  # noqa: E402
from tqdm import trange  # noqa: E402

import wandb  # noqa: E402
from src.config.config_loader import Configuration, load_config  # noqa: E402
from src.environments.factory import create_environment  # noqa: E402
from src.inference import model_simulation  # noqa: E402
from src.neural_networks.neural_network import (  # noqa: E402
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from src.replay_buffer import ReplayBuffer  # noqa: E402
from src.training import NeuralNetworkManager  # noqa: E402
from src.training_data_generator import (  # noqa: E402
    TrainingDataGenerator,
    save_training_data,
)


@torch.no_grad()
def generate_training_data(
    repr_net: RepresentationNetwork,
    dyn_net: DynamicsNetwork,
    pred_net: PredictionNetwork,
    config: Configuration,
    device: DeviceLikeType = "cpu",
    training_steps: int = 0,
    save_episodes: bool = True,
) -> None:
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

    wandb.log(
        {
            "epsilon": training_data_generator._calculate_epsilon(training_steps),
        }
    )
    if save_episodes:
        path = save_training_data(episodes)
        print(f"Training data saved to {path}")
    return episodes


def train_model(
    repr_net: RepresentationNetwork,
    dyn_net: DynamicsNetwork,
    pred_net: PredictionNetwork,
    config: Configuration,
    replay_buffer: ReplayBuffer,
    device: DeviceLikeType = "cpu",
) -> tuple[RepresentationNetwork, DynamicsNetwork, PredictionNetwork]:
    """
    Train the model using the training data.
    """

    nnm = NeuralNetworkManager(config.training, repr_net, dyn_net, pred_net, device)

    final_loss = nnm.train(replay_buffer)
    return nnm.save_models(final_loss, config.environment, False)


def generate_train_model_loop(
    n: int, config: Configuration
) -> tuple[RepresentationNetwork, DynamicsNetwork, PredictionNetwork]:
    device = torch.device("cuda" if config.runtime.use_cuda and torch.cuda.is_available() else "cpu")
    env = create_environment(config.environment, device)

    num_actions = len(env.get_action_space())
    latent_shape = config.networks.latent_shape
    repr_net = RepresentationNetwork(
        observation_space=env.get_observation_space(),
        latent_shape=latent_shape,
        config=config.networks.representation,
    ).to(device)

    dyn_net = DynamicsNetwork(
        latent_shape=latent_shape,
        num_actions=num_actions,
        config=config.networks.dynamics,
    ).to(device)

    pred_net = PredictionNetwork(
        latent_shape=latent_shape,
        num_actions=num_actions,
        config=config.networks.prediction,
    ).to(device)

    wandb.watch(repr_net, log="all")
    wandb.watch(dyn_net, log="all")
    wandb.watch(pred_net, log="all")
    replay_buffer = ReplayBuffer(config.training.replay_buffer_size, config.training.batch_size, config.training.alpha)
    for i in trange(n):
        episodes = generate_training_data(repr_net, dyn_net, pred_net, config, device, i, True)
        replay_buffer.add_episodes(episodes)

        repr_net, dyn_net, pred_net = train_model(repr_net, dyn_net, pred_net, config, replay_buffer, device)

        # Check performance of the model.
        total_reward = 0
        val_simulation_count = config.validation.simulation_count
        for _ in range(val_simulation_count):
            simulation_video_path = f"training_runs/simulation_{i:03}.mp4"

            total_reward += model_simulation(
                env,
                repr_net=repr_net,
                dyn_net=dyn_net,
                pred_net=pred_net,
                mcts_config=config.training_data_generator.mcts,
                device=device,
                inference_simulation_depth=config.validation.simulation_depth,
                human_mode=False,
                video_path=simulation_video_path,
            )
        wandb.log({"reward": total_reward / val_simulation_count, "full_loop_iteration": i})
        if i % config.validation.video_upload_interval == 0:
            wandb.log({f"Simulation_{i}": wandb.Video(simulation_video_path, caption=f"Simulation of model {i}")})

    return repr_net, dyn_net, pred_net


def _profile_code(func: Callable, *args, **kwargs) -> None:
    """
    Profile the code using cProfile.
    """
    import cProfile
    import pstats

    profile = cProfile.Profile()
    profile.enable()

    func(*args, **kwargs)

    profile.disable()

    stats = pstats.Stats(profile)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats()
    # Write the profile to a file.
    stats.dump_stats("profile.prof")


if __name__ == "__main__":
    config = load_config("config_flappy_bird.yaml")
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project=f"muzero - {config.project_name}",
        mode="disabled",
        # Track hyperparameters and run metadata.
        config=config,
    )

    # _profile_code(generate_training_data)
    # _profile_code(train_model)
    generate_train_model_loop(1000, config)

    wandb.finish()
