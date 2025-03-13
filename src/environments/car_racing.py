from typing import Any
from torch import Tensor
from gym.envs.box2d.car_racing import CarRacing as CarRacingGym
from gym.spaces.discrete import Discrete
from pydantic import BaseModel
import torch

from src.environment import Environment


class CarRacingConfig(BaseModel):
    seed: int = 42
    render_mode: str = "rgb_array"


class CarRacing(Environment):
    def __init__(self, config: CarRacingConfig):
        # By default, gym's CarRacing has continuous actions,
        # but you set continuous=False to get discrete actions.
        self.env = CarRacingGym(render_mode=config.render_mode, continuous=False)
        obs, _ = self.env.reset(seed=config.seed)  # returns (obs, info)
        # Convert the shape from (height, width, channels) to (channels, height, width)
        self.last_obs = torch.from_numpy(obs).float().permute(2, 0, 1)
        self.observation_space = self.env.observation_space

    def get_action_space(self) -> tuple[int, ...]:
        space: Discrete = self.env.action_space
        return tuple(range(space.n))

    def get_observation_space(self) -> tuple[int, ...]:
        """
        Return the shape of the observation space in the format (height, width, channels)
        The observation space is a 3D tensor of shape (96, 96, 3)
        """
        return (
            self.observation_space.shape[2],
            self.observation_space.shape[0],
            self.observation_space.shape[1],
        )

    def step(self, action: int) -> tuple[Tensor, float, bool]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Convert shape (96, 96, 3) -> (3, 96, 96)
        obs_t = torch.from_numpy(obs).float().permute(2, 0, 1)
        obs_t = obs_t.unsqueeze(0)  # (1, 3, 96, 96)
        self.last_obs = obs_t
        done = terminated or truncated
        return obs_t, reward, done

    def get_state(self) -> Tensor:
        """
        Return the last stored observation in shape (1, 96, 96, 3) NHWC format
        """
        return self.last_obs

    def reset(self) -> Tensor:
        obs, _ = self.env.reset()
        # (3, 96, 96)
        obs_t = torch.from_numpy(obs).float().permute(2, 0, 1)
        obs_t = obs_t.unsqueeze(0)  # (1, 3, 96, 96)
        self.last_obs = obs_t
        return obs_t

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
