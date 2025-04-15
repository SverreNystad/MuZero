import os
from collections.abc import Callable

import torch
from dotenv import load_dotenv
from torch._prims_common import DeviceLikeType
from tqdm import trange

import wandb
from src.config.config_loader import Configuration, load_config
from src.environments.factory import create_environment
from src.inference import model_simulation
from src.neural_networks.neural_network import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from src.replay_buffer import ReplayBuffer
from src.training import NeuralNetworkManager
from src.training_data_generator import TrainingDataGenerator, save_training_data


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
    replay_buffer = ReplayBuffer(config.training.replay_buffer_size, config.training.batch_size)
    for i in trange(n):
        episodes = generate_training_data(repr_net, dyn_net, pred_net, config, device, i, False)
        replay_buffer.add_episodes(episodes)

        repr_net, dyn_net, pred_net = train_model(repr_net, dyn_net, pred_net, config, replay_buffer, device)

        # Check performance of the model.
        total_reward = 0
        k = 3
        for _ in range(k):
            simulation_video_path = f"training_runs/simulation_{i}.mp4"
            total_reward += model_simulation(
                env,
                repr_net=repr_net,
                pred_net=pred_net,
                inference_simulation_depth=300,
                human_mode=False,
                video_path=simulation_video_path,
            )
        wandb.log({"reward": total_reward / k, "full_loop_iteration": i})
        if i % 10 == 0:
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
    config = load_config("config.yaml")
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project="muzero",
        mode="disabled",
        # Track hyperparameters and run metadata.
        config=config,
    )

    # _profile_code(generate_training_data)
    # _profile_code(train_model)
    config_name: str = "config.yaml"
    config = load_config(config_name)
    generate_train_model_loop(1000, config)

    wandb.finish()
