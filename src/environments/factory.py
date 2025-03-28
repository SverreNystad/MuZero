from src.environment import Environment
from src.environments.car_racing import CarRacing, CarRacingConfig
from src.environments.connect_four import ConnectFour, ConnectFourConfig


def create_environment(env_config: CarRacingConfig | ConnectFourConfig) -> Environment:
    match env_config:
        case CarRacingConfig():
            return CarRacing(env_config)
        case ConnectFourConfig():
            return ConnectFour(env_config)
        case _:
            raise ValueError("Invalid environment configuration")
