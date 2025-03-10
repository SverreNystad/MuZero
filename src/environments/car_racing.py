from torch import from_numpy, Tensor
from gym.envs.box2d.car_racing import CarRacing as CarRacingGym
from gym.spaces.discrete import Discrete
from pydantic import BaseModel

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
        self.last_obs = obs  # store the initial observation
        self.observation_space = self.env.observation_space

    def get_action_space(self) -> tuple:
        """
        Return a tuple of discrete actions if the environment is in discrete mode.
        """
        space: Discrete = self.env.action_space
        # For example, if space.n == 5, return (0,1,2,3,4)
        return tuple(range(space.n))

    def get_observation_space(self) -> tuple:
        return self.observation_space.shape

    def step(self, action: int) -> tuple[Tensor, float, bool]:
        """
        Steps the environment using the selected action index.
        Returns (observation, reward, termination).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs  # update the stored observation
        return from_numpy(obs), reward, terminated or truncated

    def get_state(self) -> Tensor:
        """
        Return the last stored observation as a Torch tensor.
        """
        return from_numpy(self.last_obs)

    def reset(self) -> Tensor:
        """
        Reset the CarRacing environment and store the new initial observation.
        """
        obs, _ = self.env.reset()
        self.last_obs = obs
        return from_numpy(obs)

    def render(self) -> any:
        return self.env.render()

    def close(self) -> any:
        return self.env.close()
