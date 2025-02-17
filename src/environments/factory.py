from typing import Any, Union
from src.environment import Environment


def create_environment(env_config: Union[Any]) -> Environment:

    match env_config:
        # TODO: add more cases here
        # case SpecificEnvironmentConfig:
        #   return SpecificEnvironment(env_config)
        case _:
            raise ValueError("Invalid environment configuration")
