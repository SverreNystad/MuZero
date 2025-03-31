from src.config.config_loader import load_config
from src.environment import Environment
from src.environments.factory import create_environment
from src.neural_networks.neural_network import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from src.inference import inference_simulation
config = load_config("config.yaml")
config.environment.render_mode = "human"
env = create_environment(config.environment)

inference_simulation_depth = 1000
num_actions = len(env.get_action_space())
latent_shape = config.networks.latent_shape

repr_net = RepresentationNetwork(
    observation_space=env.get_observation_space(),
    latent_shape=latent_shape,
    config=config.networks.representation,
)

pred_net = PredictionNetwork(
    latent_shape=latent_shape,
    num_actions=num_actions,
    config=config.networks.prediction,
)

inference_simulation(env, repr_net, pred_net, inference_simulation_depth)