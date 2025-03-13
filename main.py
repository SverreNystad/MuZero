from src.environments.factory import create_environment
from src.neural_network import RepresentationNetwork, DynamicsNetwork, PredictionNetwork
from src.training_data_generator import (
    TrainingDataGenerator,
    save_training_data,
)

from src.config.config_loader import load_config


def generate_training_data() -> None:
    """
    Generate training data for the training loop.
    """
    # Load the configuration file.
    config = load_config("config.yaml")

    # Create the environment using the factory method.
    env = create_environment(config["environment"])

    channels, height, width = env.get_observation_space()
    num_actions = len(env.get_action_space())
    latent_dim = config["networks"]["latent_dim"]
    # Load the representation network.
    repr_net = RepresentationNetwork(
        input_channels=channels,
        observation_space=(height, width),
        latent_dim=latent_dim,
    )

    # Load the dynamics network.
    dyn_net = DynamicsNetwork(
        latent_dim=latent_dim,
        num_actions=num_actions,
    )

    # Load the prediction network.
    pred_net = PredictionNetwork(
        latent_dim=latent_dim,
        num_actions=num_actions,
    )

    # Create the training data generator.
    training_data_generator = TrainingDataGenerator(
        env=env,
        repr_net=repr_net,
        dyn_net=dyn_net,
        pred_net=pred_net,
        config=config["training_data_generator"],
    )

    # Generate the training data.
    episodes = training_data_generator.generate_training_data()

    # Save the training data.
    path = save_training_data(episodes)
    print(f"Training data saved to {path}")


if __name__ == "__main__":
    generate_training_data()
