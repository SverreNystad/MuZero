import pytest

from src.environment import Environment
from src.environments.factory import create_environment
from src.environments.car_racing import CarRacingConfig
from src.environments.connect_four import ConnectFourConfig
from src.nerual_networks.neural_network import RepresentationNetwork
from tests.nerual_networks.test_networks import tiny_repr_net


@pytest.mark.parametrize(
    "env_config",
    [
        CarRacingConfig(seed=42, render_mode="rgb_array"),  # (3, 96, 96)
        ConnectFourConfig(),  # (2, 6, 7)
    ],
)
def test_environments_states_with_representation_network(env_config: CarRacingConfig):
    env: Environment = create_environment(env_config)

    latent_shape = (2, 1, 1)
    repr_net = tiny_repr_net(
        latent_shape=latent_shape, observation_space=env.get_observation_space()
    )

    # Test the representation network with a random observation
    obs = env.reset()
    latent = repr_net(obs)
    assert latent.shape == (1, *latent_shape)

    # Test the representation network with a random action
    action_space = env.get_action_space()
    action = action_space[0]
    obs, _, _ = env.step(action)
    next_latent = repr_net(obs)
    assert next_latent.shape == (1, *latent_shape)
    assert not (latent == next_latent).all()
