import torch

from src.config.config_loader import load_config
from src.environment import Environment
from src.environments.factory import create_environment
from src.neural_networks.neural_network import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)

def inference_simulation(env: Environment, repr_net: RepresentationNetwork, pred_net: PredictionNetwork, inference_simulation_depth: int) -> None:
    state = env.get_state()
    for i in range(inference_simulation_depth):
        # Get the current state of the environment.
        env.render()
        # Encode the state using the representation network.
        latent_state = repr_net(state)  # add batch dimension

        # add batch dimension
        latent_state = latent_state

        policy, value = pred_net(latent_state)
        
        # Pick the action with the highest probability.
        action = torch.argmax(policy).item()
        print(f"Action: {action}")
        
        # Step the environment using the action.
        state, reward, done = env.step(action)
       
        # Check if the episode is done.
        if done:
            break
