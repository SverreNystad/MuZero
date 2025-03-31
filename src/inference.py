import torch

from src.environment import Environment
from src.neural_networks.neural_network import (
    PredictionNetwork,
    RepresentationNetwork,
)


def model_simulation(env: Environment, repr_net: RepresentationNetwork, pred_net: PredictionNetwork, inference_simulation_depth: int) -> None:
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
        
        # Step the environment using the action.
        state, reward, done = env.step(action)
        # Print the action, reward, and value.
        print(f"Step {i}: Action: {action}, Reward: {reward}, Value: {value.item()}")
       
        # Check if the episode is done.
        if done:
            break
