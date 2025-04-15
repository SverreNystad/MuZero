import optuna

import wandb
from main import generate_train_model_loop
from src.config.config_loader import (
    TrainingConfig,
    TrainingDataGeneratorConfig,
    load_config,
)
from src.environments.factory import create_environment
from src.inference import model_simulation


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
    repr_net, dyn_net, pred_net = generate_train_model_loop(10, config)
    env = create_environment(config.environment)
    running_reward = model_simulation(
        env,
        repr_net=repr_net,
        pred_net=pred_net,
        inference_simulation_depth=1000,
        human_mode=False,
    )

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
