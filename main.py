from collections.abc import Callable

from torch._prims_common import DeviceLikeType
from tqdm import trange
import torch
import wandb
import os
import optuna
from dotenv import load_dotenv

from src.config.config_loader import Configuration, TrainingConfig, TrainingDataGeneratorConfig, load_config
from src.environments.factory import create_environment
from src.neural_networks.neural_network import (
    DynamicsNetwork,
    PredictionNetwork,
    RepresentationNetwork,
)
from src.training import NeuralNetworkManager
from src.training_data_generator import (
    TrainingDataGenerator,
    delete_all_training_data,
    load_all_training_data,
    save_training_data,
)
from src.inference import model_simulation


@torch.no_grad()
def generate_training_data(
    repr_net: RepresentationNetwork | None,
    dyn_net: DynamicsNetwork | None,
    pred_net: PredictionNetwork | None,
    config: Configuration,
    device: DeviceLikeType = "cpu",
) -> None:
    """
    Generate training data for the training loop.
    """
    # Create the environment using the factory method.
    env = create_environment(config.environment, device)

    num_actions = len(env.get_action_space())
    latent_shape = config.networks.latent_shape
    # Load the representation network.
    if repr_net is None:
        repr_net = RepresentationNetwork(
            observation_space=env.get_observation_space(),
            latent_shape=latent_shape,
            config=config.networks.representation,
        ).to(device)

    # Load the dynamics network.
    if dyn_net is None:
        dyn_net = DynamicsNetwork(
            latent_shape=latent_shape,
            num_actions=num_actions,
            config=config.networks.dynamics,
        ).to(device)

    # Load the prediction network.
    if pred_net is None:
        pred_net = PredictionNetwork(
            latent_shape=latent_shape,
            num_actions=num_actions,
            config=config.networks.prediction,
        ).to(device)

    # Create the training data generator.
    training_data_generator = TrainingDataGenerator(
        env=env,
        repr_net=repr_net,
        dyn_net=dyn_net,
        pred_net=pred_net,
        config=config.training_data_generator,
    )

    # Generate the training data.
    episodes = training_data_generator.generate_training_data()

    # Save the training data.
    path = save_training_data(episodes)
    print(f"Training data saved to {path}")


def train_model(
    repr_net: RepresentationNetwork | None,
    dyn_net: DynamicsNetwork | None,
    pred_net: PredictionNetwork | None,
    config: Configuration,
    device: DeviceLikeType = "cpu",
) -> tuple[RepresentationNetwork, DynamicsNetwork, PredictionNetwork]:
    """
    Train the model using the training data.
    """
    # load episodes
    episodes = load_all_training_data()

    # load networks
    # Create the environment using the factory method.
    env = create_environment(config.environment, device)

    num_actions = len(env.get_action_space())
    latent_shape = config.networks.latent_shape
    # Load the representation network.
    if repr_net is None:
        repr_net = RepresentationNetwork(
            observation_space=env.get_observation_space(),
            latent_shape=latent_shape,
            config=config.networks.representation,
        ).to(device)

    # Load the dynamics network.
    if dyn_net is None:
        dyn_net = DynamicsNetwork(
            latent_shape=latent_shape,
            num_actions=num_actions,
            config=config.networks.dynamics,
        ).to(device)

    # Load the prediction network.
    if pred_net is None:
        pred_net = PredictionNetwork(
            latent_shape=latent_shape,
            num_actions=num_actions,
            config=config.networks.prediction,
        ).to(device)

    nnm = NeuralNetworkManager(config.training, repr_net, dyn_net, pred_net)

    # train using episodes
    final_loss = nnm.train(episodes)
    return nnm.save_models(final_loss, config.environment, False)


def generate_train_model_loop(n: int, config: Configuration) -> tuple[RepresentationNetwork, DynamicsNetwork, PredictionNetwork]:
    repr_net = None
    dyn_net = None
    pred_net = None
    device = torch.device(
        "cuda" if config.runtime.use_cuda and torch.cuda.is_available() else "cpu"
    )
    for _ in trange(n):
        generate_training_data(repr_net, dyn_net, pred_net, config)
        repr_net, dyn_net, pred_net = train_model(repr_net, dyn_net, pred_net, config)
        # As better models create better training data, we can delete the old training data.
        delete_all_training_data()

    return repr_net, dyn_net, pred_net

def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna.
    """
    # Define the hyperparameters to optimize.
    config_name: str = "config.yaml"
    config = load_config(config_name)
    config.training_data_generator = TrainingDataGeneratorConfig(
        num_episodes=trial.suggest_int("num_episodes", 1, 2),
        max_steps_per_episode=trial.suggest_int("max_steps", 1, 2),
        total_time=3600,
        mcts=config.training_data_generator.mcts,
    )
    config.training = TrainingConfig(
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        batch_size=trial.suggest_int("batch_size", 1, 256),
        epochs=trial.suggest_int("epochs", 1, 100), 
        betas=(0.9, 0.999),
        roll_ahead=trial.suggest_int("roll_ahead", 1, 10),
        look_back=trial.suggest_int("look_back", 1, 10),
        mini_batch_size=trial.suggest_int("mini_batch_size", 1, 256),
    )

    # Run the training loop.
    repr_net, dyn_net, pred_net = generate_train_model_loop(1, config)
    env = create_environment(config.environment)
    running_reward = model_simulation(env, repr_net=repr_net, pred_net=pred_net, inference_simulation_depth=1000, human_mode=False)

    return running_reward
    
def hyperparameter_search(n_trials: int) -> tuple[dict, float]:

    # Perform hyperparameter search using Optuna.
    study_name = "optuna_training_and_data_generator_tuning"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        direction="maximize",
        storage=storage_name,
    )
    study.optimize(objective, n_trials=n_trials)
    # Save the best hyperparameters.
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    wandb.log(
        {
            "best_params": best_params,
            "best_value": best_value,
        }
    )
    return best_params, best_value

def _profile_code(func: Callable, *args, **kwargs) -> None:
    """
    Profile the code using cProfile.
    """
    import cProfile
    import pstats

    profile = cProfile.Profile()
    profile.enable()

    func(*args, **kwargs)

    profile.disable()

    stats = pstats.Stats(profile)
    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats()
    # Write the profile to a file.
    stats.dump_stats("profile.prof")


if __name__ == "__main__":
    config = load_config("config.yaml")
    load_dotenv()
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project="muzero",
        # Track hyperparameters and run metadata.
    )

    # _profile_code(generate_training_data)
    # _profile_code(train_model)
    config_name: str = "config.yaml"
    config = load_config(config_name)
    hyperparameter_search(2)