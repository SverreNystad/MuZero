import pytest
from src.environments.factory import create_environment
from src.environment import Environment


def test_create_non_valid_environment():
    with pytest.raises(ValueError):
        create_environment("invalid_config")


# TODO: Create more tests for the create_environment function that uses correct configurations objects
