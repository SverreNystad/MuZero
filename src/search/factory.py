from enum import StrEnum

from pydantic import BaseModel
import torch

from src.search.mcts import MCTS
from src.search.selection import UCT, PUCT
from src.search.backpropagation import Backpropagation
from src.neural_network import DynamicsNetwork, PredictionNetwork
from src.search.simulation import MuZeroSimulation
from src.search.strategies import SelectionStrategy


class SelectionStrategyType(StrEnum):
    uct = "uct"
    puct = "puct"


class MCTSConfig(BaseModel):
    selection_strategy: SelectionStrategyType = SelectionStrategyType.puct
    max_iterations: int
    max_time: int


def create_mcts(
    dynamics_network: DynamicsNetwork,
    prediction_network: PredictionNetwork,
    actions: torch.Tensor,
    config: MCTSConfig,
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
    simulation_strategy = MuZeroSimulation(prediction_network)
    backpropagation_strategy = Backpropagation()

    # Instantiate the MCTS object with the given strategies.
    mcts_instance = MCTS(
        selection=selection_strategy,
        simulation=simulation_strategy,
        backpropagation=backpropagation_strategy,
        dynamic_network=dynamics_network,
        actions=actions,
        max_itr=config.max_iterations,
        max_time=config.max_time,
    )
    return mcts_instance
