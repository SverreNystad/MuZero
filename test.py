from src.environments.connect_four import ConnectFour, ConnectFourConfig
import gym

env = gym.make("CarRacing-v2", render_mode="human")
env.reset(seed=42)

env.render()
