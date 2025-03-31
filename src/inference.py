import torch

from src.environment import Environment
from src.neural_networks.neural_network import (
    PredictionNetwork,
    RepresentationNetwork,
)


def model_simulation(env: Environment, repr_net: RepresentationNetwork, pred_net: PredictionNetwork, inference_simulation_depth: int, human_mode: bool = True) -> None:
    env.reset()
    state = env.get_state()
    running_reward = 0.0
    for i in range(inference_simulation_depth):
        # Get the current state of the environment.
        if human_mode:
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
        running_reward += reward
        # Print the action, reward, and value.
        if human_mode:
            print(f"Step {i}: Action: {action}, Reward: {reward}, Value: {value.item()}")
       
        # Check if the episode is done.
        if done:
            return running_reward
    return running_reward