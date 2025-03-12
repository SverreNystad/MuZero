import pytest

from src.environment import Environment
from src.environments.factory import create_environment
from src.environments.car_racing import CarRacingConfig
from src.environments.connect_four import ConnectFourConfig
from src.neural_network import RepresentationNetwork


@pytest.mark.parametrize(
    "env_config",
    [
        CarRacingConfig(seed=42, render_mode="rgb_array"),  # (3, 96, 96)
    ],
)
def test_environments_states_with_representation_network(env_config: CarRacingConfig):
    env: Environment = create_environment(env_config)
    latent_dim = 32
    channels, height, width = env.get_observation_space()
    repr_net = RepresentationNetwork(channels, (height, width), latent_dim)

    # Test the representation network with a random observation
    obs = env.reset()
    latent = repr_net(obs)
    assert latent.shape == (1, latent_dim)
