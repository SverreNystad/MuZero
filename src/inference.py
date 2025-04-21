import imageio
import torch
from torch import Tensor
from tqdm import trange

from src.config.config_loader import MCTSConfig
from src.environment import Environment
from src.neural_networks.neural_network import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from src.ring_buffer import Frame, FrameRingBuffer, make_history_tensor
from src.search.factory import create_mcts
from src.search.mcts import MCTS
from src.search.nodes import Node
from src.training_data_generator import _make_actions_tensor


def model_simulation(
    env: Environment,
    repr_net: RepresentationNetwork,
    dyn_net: DynamicsNetwork,
    pred_net: PredictionNetwork,
    inference_simulation_depth: int,
    mcts_config: MCTSConfig,
    device: str = "cpu",
    human_mode: bool = True,
    video_path: str = "simulation.mp4",
) -> float:
    print("Starting model simulation...")
    env.reset()
    if human_mode:
        env.env.render_mode = "human"

    state = env.get_state()
    mcts: MCTS = create_mcts(
        dynamics_network=dyn_net,
        prediction_network=pred_net,
        actions=_make_actions_tensor(env, device),
        config=mcts_config,
    )
    running_reward = 0.0
    frames = []

    ringbuffer = FrameRingBuffer(repr_net.history_length)
    ringbuffer.fill(Frame(state, 0))

    for i in trange(inference_simulation_depth):
        # Get the current state of the environment.
        frame = env.render()
        frames.append(frame)

        # Encode the state using the representation network.
        history_tensor = make_history_tensor(ringbuffer).to(device)

        latent_state = repr_net(history_tensor)  # (batch, channels, height, width) -> (1, (3 + 1) * 32, 512, 288)
        # Run MCTS to get policy distribution (tree_policy) and value estimate.
        root = Node(latent_state=latent_state, to_play=env.get_to_play())
        tree_policy, value = mcts.run(root)
        policy_tensor = Tensor(tree_policy).to(device)

        # Pick the action with the highest probability.
        action = torch.argmax(policy_tensor).item()

        # Step the environment using the action.
        state, reward, done = env.step(action)
        running_reward += reward

        ringbuffer.add(Frame(state, action))

        if human_mode:
            print(f"Step {i}: Action: {action}, Reward: {reward}, Value: {value.item()}")

        # Check if the episode is done.
        if done:
            break

    # Save the frames as a GIF.
    # Note: need to set the macro_block_size to None to avoid a warning.
    print(f"Saving video to {video_path}...")
    kargs = {"macro_block_size": None, "ffmpeg_params": ["-s", "600x400"]}
    imageio.mimsave(video_path, frames, fps=30, **kargs)

    return running_reward
