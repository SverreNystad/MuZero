from collections.abc import Callable

from tqdm import trange
import torch
from src.config.config_loader import load_config
from src.environments.factory import create_environment
from src.neural_networks.neural_network import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from src.training import NeuralNetworkManager
from src.training_data_generator import (
    TrainingDataGenerator,
    delete_all_training_data,
    load_all_training_data,
    save_training_data,
)

@torch.no_grad()
def generate_training_data(
    repr_net: RepresentationNetwork | None,
    dyn_net: DynamicsNetwork | None,
    pred_net: PredictionNetwork | None,
    config_name: str = "config.yaml",
) -> None:
    """
    Generate training data for the training loop.
    """
    # Load the configuration file.
    config = load_config(config_name)

    # Create the environment using the factory method.
    env = create_environment(config.environment)

    num_actions = len(env.get_action_space())
    latent_shape = config.networks.latent_shape
    # Load the representation network.
    if repr_net is None:
        repr_net = RepresentationNetwork(
            observation_space=env.get_observation_space(),
            latent_shape=latent_shape,
            config=config.networks.representation,
        )

    # Load the dynamics network.
    if dyn_net is None:
        dyn_net = DynamicsNetwork(
            latent_shape=latent_shape,
            num_actions=num_actions,
            config=config.networks.dynamics,
        )

    # Load the prediction network.
    if pred_net is None:
        pred_net = PredictionNetwork(
            latent_shape=latent_shape,
            num_actions=num_actions,
            config=config.networks.prediction,
        )

    # Create the training data generator.
    training_data_generator = TrainingDataGenerator(
        env=env,
        repr_net=repr_net,
        dyn_net=dyn_net,
        pred_net=pred_net,
        config=config.training_data_generator,
    )

    # Generate the training data.
    episodes = training_data_generator.generate_training_data()

    # Save the training data.
    path = save_training_data(episodes)
    print(f"Training data saved to {path}")


def train_model(
    repr_net: RepresentationNetwork | None,
    dyn_net: DynamicsNetwork | None,
    pred_net: PredictionNetwork | None,
    config_name: str = "config.yaml",
) -> tuple[RepresentationNetwork, DynamicsNetwork, PredictionNetwork]:
    """
    Train the model using the training data.
    """
    # load config
    config = load_config(config_name)

    # load episodes
    episodes = load_all_training_data()

    # load networks
    # Create the environment using the factory method.
    env = create_environment(config.environment)

    num_actions = len(env.get_action_space())
    latent_shape = config.networks.latent_shape
    # Load the representation network.
    if repr_net is None:
        repr_net = RepresentationNetwork(
            observation_space=env.get_observation_space(),
            latent_shape=latent_shape,
            config=config.networks.representation,
        )

    # Load the dynamics network.
    if dyn_net is None:
        dyn_net = DynamicsNetwork(
            latent_shape=latent_shape,
            num_actions=num_actions,
            config=config.networks.dynamics,
        )

    # Load the prediction network.
    if pred_net is None:
        pred_net = PredictionNetwork(
            latent_shape=latent_shape,
            num_actions=num_actions,
            config=config.networks.prediction,
        )

    nnm = NeuralNetworkManager(config.training, repr_net, dyn_net, pred_net)

    # train using episodes
    final_loss = nnm.train(episodes)
    return nnm.save_models(final_loss, config.environment, False)


def generate_train_model_loop(n: int, config_name: str = "config.yaml") -> None:
    repr_net = None
    dyn_net = None
    pred_net = None
    for _ in trange(n):
        generate_training_data(repr_net, dyn_net, pred_net, config_name)
        repr_net, dyn_net, pred_net = train_model(repr_net, dyn_net, pred_net, config_name)
        # As better models create better training data, we can delete the old training data.
        delete_all_training_data()


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
    # _profile_code(generate_training_data)
    # _profile_code(train_model)
    generate_train_model_loop(5)
