import torch

from src.config.config_loader import MCTSConfig, SelectionStrategyType
from src.neural_networks.neural_network import DynamicsNetwork, PredictionNetwork
from src.search.backpropagation import Backpropagation
from src.search.mcts import MCTS
from src.search.selection import PUCT, UCT
from src.search.simulation import MuZeroSimulation
from src.search.strategies import SelectionStrategy


def create_mcts(
    dynamics_network: DynamicsNetwork,
    prediction_network: PredictionNetwork,
    actions: torch.Tensor,
    config: MCTSConfig,
    device="cpu",
) -> MCTS:
    """
    Factory method for creating an MCTS instance.

    Args:
        dynamics_network (DynamicsNetwork): The dynamics network used for state transitions.
        prediction_network (PredictionNetwork): The prediction network used in the simulation.
        actions (torch.Tensor): A tensor containing the set of actions.
        config (MCTSConfig): A configuration object for the MCTS instance.

    Returns:
        MCTS: A configured MCTS instance.
    """
    # Choose selection strategy based on input parameter.
    selection_strategy: SelectionStrategy
    match config.selection_strategy:
        case SelectionStrategyType.uct:
            selection_strategy = UCT()
        case SelectionStrategyType.puct:
            selection_strategy = PUCT()

    # Create simulation and backpropagation strategies.
    simulation_strategy = MuZeroSimulation(dynamics_network, prediction_network, config.model_look_ahead, device)
    backpropagation_strategy = Backpropagation(config.discount_factor)

    # Instantiate the MCTS object with the given strategies.
    mcts_instance = MCTS(
        selection=selection_strategy,
        simulation=simulation_strategy,
        backpropagation=backpropagation_strategy,
        dynamic_network=dynamics_network,
        prediction_network=prediction_network,
        actions=actions,
        max_itr=config.max_iterations,
        max_time=config.max_time,
    )
    return mcts_instance
