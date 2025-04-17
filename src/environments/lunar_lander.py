from typing import Any, Literal

import torch
from gym.envs.box2d.lunar_lander import LunarLander as LunarLanderGym
from gym.spaces.discrete import Discrete
from pydantic import BaseModel
from torch import Tensor
from torch._prims_common import DeviceLikeType

from src.environment import Environment


class LunarLanderConfig(BaseModel):
    type: Literal["LunarLander"] = "LunarLander"
    seed: int = 42
    render_mode: str = "rgb_array"


class LunarLander(Environment):
    def __init__(self, config: LunarLanderConfig, device: DeviceLikeType):
        # By default, gym's CarRacing has continuous actions,
        # but you set continuous=False to get discrete actions.
        self.env = LunarLanderGym(render_mode=config.render_mode, continuous=False)
        obs, _ = self.env.reset(seed=config.seed)  # returns (obs, info)
        self.device = device
        # Convert the shape from shape (8, ) to shape (1, 1, 8)
        self.last_obs = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 1, 8)
        self.observation_space = self.env.observation_space

        self.to_play = 0

    def get_to_play(self) -> int:
        return self.to_play

    def get_action_space(self) -> tuple[int, ...]:
        space: Discrete = self.env.action_space
        return tuple(range(space.n))

    def get_observation_space(self) -> tuple[int, ...]:
        """
        Return the shape of the observation space in the format (?, ?, obsvation data)
        The observation space is a 3D tensor of shape (1, 1, 8)
        """
        return (1, 1, 8)

    def step(self, action: int) -> tuple[Tensor, float, bool]:
        obs, reward, terminated, truncated, _ = self.env.step(action)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 1, 8)
        self.last_obs = obs_t
        done = terminated or truncated
        return obs_t, reward, done

    def get_state(self) -> Tensor:
        """
        Return the last stored observation in shape (1, 1, 1, 8) NHWC format
        """
        return self.last_obs

    def reset(self) -> Tensor:
        obs, _ = self.env.reset()
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 1, 8)
        self.last_obs = obs_t
        return obs_t

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
