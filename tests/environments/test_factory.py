import pytest

from src.config.config_loader import load_config
from src.environments.car_racing import CarRacing, CarRacingConfig
from src.environments.connect_four import ConnectFour, ConnectFourConfig
from src.environments.factory import create_environment


def test_create_non_valid_environment():
    with pytest.raises(ValueError):
        create_environment("invalid_config")


def test_create_environment_with_car_racing_config():
    env_config = CarRacingConfig()
    env = create_environment(env_config)
    assert isinstance(env, CarRacing)


def test_create_environment_with_connect_four_config():
    env_config = ConnectFourConfig()
    env = create_environment(env_config)
    assert isinstance(env, ConnectFour)


def test_config_loader_creating_environment():
    config = load_config("config.yaml")
    env = create_environment(config.environment)
    assert env is not None
