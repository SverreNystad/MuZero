from src.environments.connect_four import ConnectFour, ConnectFourConfig

config = ConnectFourConfig()
env = ConnectFour(config=config)

print(env.render())