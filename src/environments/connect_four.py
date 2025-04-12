from typing import Any, Literal

from gymnasium.spaces import Box
from pettingzoo.classic import connect_four_v3
from pydantic import BaseModel
from torch import Tensor, from_numpy
from torch._prims_common import DeviceLikeType

from src.environment import Environment


class ConnectFourConfig(BaseModel):
    type: Literal["ConnectFour"] = "ConnectFour"
    render_mode: str = "rgb_array"
    seed: int = 42


class ConnectFour(Environment):
    def __init__(self, config: ConnectFourConfig, device: DeviceLikeType):
        self.env = connect_four_v3.env(render_mode=config.render_mode)
        self.env.reset(seed=config.seed)
        self.observation_space = self.env.observation_space("player_0")
        self.device = device
        self.to_play = self.env.agent_selection

    def get_to_play(self) -> int:
        return self.to_play

    def get_action_space(self) -> tuple[int, ...]:
        space: Box = self.observation_space["action_mask"]
        # Box(0, 1, (7,), int8)
        return tuple(range(space.shape[0]))

    def get_observation_space(self) -> tuple[int, ...]:
        return (
            self.observation_space["observation"].shape[2],
            *self.observation_space["observation"].shape[:2],
        )

    def step(self, action: int) -> tuple[Tensor, float, bool]:
        self.env.step(action)
        self.to_play = self.env.agent_selection

        observation, reward, termination, _, _ = self.env.last()
        observation_t = from_numpy(observation["observation"]).to(self.device)
        observation_t = observation_t.float().permute(2, 0, 1)
        observation_t = observation_t.unsqueeze(0)

        return observation_t, reward, termination

    def get_state(self) -> Tensor:
        observation = self.env.last()[0]
        return from_numpy(observation["observation"]).float().permute(2, 0, 1).to(self.device)

    def reset(self) -> Tensor:
        self.env.reset()
        self.to_play = self.env.agent_selection

        observation = self.env.last()[0]
        obs_t = from_numpy(observation["observation"]).float().permute(2, 0, 1).to(self.device)
        obs_t = obs_t.unsqueeze(0)
        return obs_t

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
