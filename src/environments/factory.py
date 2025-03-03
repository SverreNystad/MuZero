from typing import Any, Union
from src.environment import Environment
from src.environments.connect_four import ConnectFour, ConnectFourConfig


def create_environment(env_config: Union[Any]) -> Environment:

    match env_config:
        # TODO: add more cases here
        # case SpecificEnvironmentConfig:
        #   return SpecificEnvironment(env_config)
        case ConnectFourConfig():
            return ConnectFour(env_config)
        case _:
            raise ValueError("Invalid environment configuration")
