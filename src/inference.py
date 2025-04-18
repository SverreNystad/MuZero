import imageio
import torch

from src.ring_buffer import FrameRingBuffer, Frame, make_history_tensor
from src.environment import Environment
from src.neural_networks.neural_network import PredictionNetwork, RepresentationNetwork


def model_simulation(
    env: Environment,
    repr_net: RepresentationNetwork,
    pred_net: PredictionNetwork,
    inference_simulation_depth: int,
    human_mode: bool = True,
    video_path: str = "simulation.mp4",
) -> float:
    print("Starting model simulation...")
    env.reset()
    if human_mode:
        env.env.render_mode = "human"

    state = env.get_state()
    running_reward = 0.0
    frames = []


    ringbuffer = FrameRingBuffer(repr_net.history_length)
    ringbuffer.fill(Frame(state, 0))

    for i in range(inference_simulation_depth):
        # Get the current state of the environment.
        frame = env.render()
        frames.append(frame)


        # Encode the state using the representation network.
        latent_state = repr_net(make_history_tensor(ringbuffer))

        policy, value = pred_net(latent_state)

        # Pick the action with the highest probability.
        action = torch.argmax(policy).item()

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
