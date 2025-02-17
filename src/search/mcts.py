from dataclasses import dataclass

import torch


@dataclass
class MCTSOutput:
    """
    Output of the MCTS algorithm.
    """

    action_probs: torch.Tensor
    value: torch.float
