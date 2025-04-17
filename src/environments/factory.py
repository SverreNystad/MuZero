from torch._prims_common import DeviceLikeType

from src.environment import Environment
from src.environments.car_racing import CarRacing, CarRacingConfig
from src.environments.connect_four import ConnectFour, ConnectFourConfig
from src.environments.lunar_lander import LunarLander, LunarLanderConfig


def create_environment(env_config: CarRacingConfig | ConnectFourConfig | LunarLanderConfig, device: DeviceLikeType = "cpu") -> Environment:
    match env_config:
        case LunarLanderConfig():
            return LunarLander(env_config, device)
        case CarRacingConfig():
            return CarRacing(env_config, device)
        case ConnectFourConfig():
            return ConnectFour(env_config, device)
        case _:
            raise ValueError("Invalid environment configuration")
