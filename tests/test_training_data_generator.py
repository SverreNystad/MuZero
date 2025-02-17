from src.environments.factory import create_environment
from src.training_data_generator import Episode, TrainingDataGenerator

def test_generate_training_data_gives_episode_data():
    env_config = {}
    env = create_environment(env_config)

    generator = TrainingDataGenerator(env)

    training_data = generator.generate_training_data(1, 30_000)

    assert len(training_data) == 1
    assert isinstance(training_data, list[Episode])