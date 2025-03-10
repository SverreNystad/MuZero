from torch import from_numpy, Tensor

from gym.envs.box2d.car_racing import CarRacing as CarRacingGym
from gym.spaces.discrete import Discrete
from pydantic import BaseModel

from src.environment import Environment


class CarRacingConfig(BaseModel):
    seed: int = 42


class CarRacing(Environment):
    def __init__(self, config: CarRacingConfig):
        self.env = CarRacingGym(render_mode="human+", continuous=False)
        self.env.reset(seed=config.seed)
        self.observation_space = self.env.observation_space

    def get_action_space(self) -> tuple:
        space: Discrete = self.env.action_space
        return space.start, space.n

    def get_observation_space(self) -> tuple:
        return self.observation_space.shape

    def step(self, action: int) -> tuple[Tensor, float, bool]:
        state = self.env.step(action)
        observation, reward, termination, truncated, info = state
        return from_numpy(observation), reward, termination

    def get_state(self) -> any:
        return from_numpy(self.env.last()[0])

    def reset(self) -> any:
        return self.env.reset()

    def render(self) -> any:
        return self.env.render()

    def close(self) -> any:
        return self.env.close()
