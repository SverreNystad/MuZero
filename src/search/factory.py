import torch
from src.search.mcts import MCTS
from src.search.selection import UCT, PUCT
from src.search.backpropagation import Backpropagation
from src.neural_network import DynamicsNetwork, PredictionNetwork
from src.search.simulation import MuZeroSimulation
from src.search.strategies import SelectionStrategy


def create_mcts(
    dynamics_network: DynamicsNetwork,
    prediction_network: PredictionNetwork,
    actions: torch.Tensor,
    selection_type: str = "uct",  # or "puct"
    max_itr: int = 0,
    max_time: float = 10.0,
) -> MCTS:
    """
    Factory method for creating an MCTS instance.

    Args:
        dynamics_network (DynamicsNetwork): The dynamics network used for state transitions.
        prediction_network (PredictionNetwork): The prediction network used in the simulation.
        actions (torch.Tensor): A tensor containing the set of actions.
        selection_type (str, optional): The type of selection strategy to use ('uct' or 'puct').
                                        Defaults to 'uct'.
        max_itr (int, optional): The maximum number of iterations for MCTS. Set to 0 to use time-based termination.
        max_time (float, optional): The maximum time (in seconds) to run MCTS if max_itr is 0.

    Returns:
        MCTS: A configured MCTS instance.
    """
    # Choose selection strategy based on input parameter.
    selection_strategy: SelectionStrategy
    if selection_type.lower() == "uct":
        selection_strategy = UCT()
    elif selection_type.lower() == "puct":
        selection_strategy = PUCT()
    else:
        raise ValueError(
            f"Invalid selection_type: {selection_type}. Choose 'uct' or 'puct'."
        )

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
        max_itr=max_itr,
        max_time=max_time,
    )
    return mcts_instance
