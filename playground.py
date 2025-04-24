import pygame
from src.config.config_loader import load_config
from src.environments.factory import create_environment


config = load_config("config_flappy_bird.yaml")
config.environment.render_mode = "human"
env = create_environment(config.environment)

print(env.get_state().shape)

inference_simulation_depth = 1000
num_actions = len(env.get_action_space())
# latent_shape = config.networks.latent_shape


def demo_flappy():
    # Initialize the environment

    while True:
        env.reset()
        done = False
        step = 0

        cum_reward = 0

        while not done:
            step += 1
            # Sample a random action

            # action = random.choice(env.get_action_space())
            key = pygame.key.get_pressed()
            if key[pygame.K_SPACE]:
                action = 1
            else:
                action = 0

            # Step the environment
            _, reward, done = env.step(action)

            cum_reward += reward
            # Render the environment
            env.render()

        print(cum_reward)

    # Close the environment
    env.close()

if __name__ == "__main__":
    demo_flappy()