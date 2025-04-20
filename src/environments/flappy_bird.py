from typing import Any, Literal

import torch
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdEnv
from gym.spaces.discrete import Discrete
from pydantic import BaseModel
from torch import Tensor
from torch._prims_common import DeviceLikeType

from src.environment import Environment


class FlappyBirdConfig(BaseModel):
    type: Literal["FlappyBird"] = "FlappyBird"
    seed: int = 42
    render_mode: str = "rgb_array"


class FlappyBird(Environment):
    def __init__(self, config: FlappyBirdConfig, device: DeviceLikeType):
        # By default, gym's CarRacing has continuous actions,
        # but you set continuous=False to get discrete actions.
        self.env = FlappyBirdEnv(render_mode=config.render_mode, use_lidar=False)
        self.device = device
        # Convert the shape from shape (512, 288, 3) to shape (3, 512, 288)
        self.last_obs = torch.Tensor()
        self.observation_space = self.get_observation_space()

        self.to_play = 0

        self.reset()

    def get_to_play(self) -> int:
        return self.to_play

    def get_action_space(self) -> tuple[int, ...]:
        space: Discrete = self.env.action_space
        return tuple(range(space.n))

    def get_observation_space(self) -> tuple[int, ...]:
        """
        Return the shape of the observation space in the format (height, width, channels)
        The observation space is a 3D tensor of shape (512, 288, 3)
        """
        return (3, 512, 288)

    def step(self, action: int) -> tuple[Tensor, float, bool]:
        step = self.env.step(action)
        _, reward, done, _ = step[0], step[1], step[2], step[3]
        obs_t = torch.from_numpy(self.render()).float().permute(2, 0, 1).to(self.device) # (3, 512, 288)
        obs_t = obs_t.unsqueeze(0) # (1, 3, 512, 288)
        self.last_obs = obs_t
        return obs_t, reward, done

    def get_state(self) -> Tensor:
        """
        Return the last stored observation in shape (1, 96, 96, 3) NHWC format
        """
        return self.last_obs

    def reset(self) -> Tensor:
        _, _ = self.env.reset()
        obs_t = torch.from_numpy(self.render()).float().permute(2, 0, 1).to(self.device)
        obs_t = obs_t.unsqueeze(0)  # (1, 3, 512, 288)
        self.last_obs = obs_t
        return obs_t

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
