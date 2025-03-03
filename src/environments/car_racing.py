import gym

env = gym.make("CarRacing-v0")
env.reset(seed=42)

env.render()

