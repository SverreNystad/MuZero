from src.environment import Environment
from src.environments.factory import create_environment
from src.environments.car_racing import CarRacing, CarRacingConfig
from src.neural_network import RepresentationNetwork


def test_car_racing_with_representation_network():
    env: CarRacing = create_environment(CarRacingConfig())
    latent_dim = 32
    channels, height, width = env.get_observation_space()
    repr_net = RepresentationNetwork(channels, (height, width), latent_dim)

    # Test the representation network with a random observation
    obs = env.reset()  # (3, 96, 96)
    latent = repr_net(obs)
    assert latent.shape == (1, latent_dim)
