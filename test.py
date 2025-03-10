from src.environments.connect_four import ConnectFour, ConnectFourConfig
from src.environments.car_racing import CarRacing, CarRacingConfig

env = CarRacing(CarRacingConfig())
print(env.get_observation_space())
print(env.get_action_space())
