
from torch import from_numpy, Tensor
from pydantic import BaseModel
from pettingzoo.classic import connect_four_v3
from src.environment import Environment
from gymnasium.spaces import Box


env = connect_four_v3.env(render_mode="rgb_array")
env.reset(seed=42)



class ConnectFour(Environment):
    def __init__(self):
        self.env = connect_four_v3.env(render_mode="rgb_array")
        self.env.reset(seed=42)
        self.observation_space = self.env.observation_space("player_0")

    def get_action_space(self) -> tuple:
        return self.observation_space["action_mask"].shape
    
    def get_observation_space(self) -> tuple:
        return self.observation_space["observation"].shape
    
    def step(self, action: int) -> tuple:
        self.env.step(action)
        observation, reward, termination, truncation, info = self.env.last()
        observation = from_numpy(observation["observation"])
        return observation, reward, termination

    def get_state(self) -> Tensor:
        observation = self.env.last()[0]
        return from_numpy(observation["observation"])
    
    def reset(self) -> Tensor:
        self.env.reset()
        observation = self.env.last()[0]
        return from_numpy(observation["observation"])
    
    def render(self, mode: str = "human") -> any:
        return self.env.render()
    
