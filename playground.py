from src.config.config_loader import load_config
from src.environments.factory import create_environment
from src.inference import model_simulation
from src.neural_networks.neural_network import load_networks

config = load_config("config.yaml")
config.environment.render_mode = "human"
env = create_environment(config.environment)

inference_simulation_depth = 1000
num_actions = len(env.get_action_space())
latent_shape = config.networks.latent_shape

model_folder_path = "training_runs/models/6_20250331_103553"
observation_space = env.get_observation_space()
num_actions = len(env.get_action_space())
repr_net, dyn_net, pred_net = load_networks(model_folder_path)

running_reward = model_simulation(env, repr_net, pred_net, inference_simulation_depth)
print(f"Running reward: {running_reward}")
