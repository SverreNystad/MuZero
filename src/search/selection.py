from math import log, sqrt
from typing import cast

from src.search.nodes import Node
from src.search.strategies import SelectionStrategy


class UCT(SelectionStrategy):
    """
    Upper Confidence bounds applied to Trees
    """

    def __init__(self, c: float = 1.41):
        self.c = c

    def uct_score(self, node: Node) -> float:
        prior_score = node.reward / node.visit_count
        parent = cast(Node, node.parent)
        value_score = self.c * sqrt(log(parent.visit_count)) / (node.visit_count + 1)
        return prior_score + value_score

    def __call__(self, root: Node) -> Node:
        while len(root.children.values()) > 0:
            root = max(root.children.values(), key=self.uct_score)
        return root


class PUCT(SelectionStrategy):
    """
    Predictor Upper Confidence Bound applied to Trees
    """

    def __init__(self, c1: float = 1.25, c2: float = 19652, discount: float = 0.99):
        """
        Args:
            - c1 (float): The PUCT-initializer parameter set as 1.25 as (Silver et al., 2017) suggests.
            - c2 (float): The PUCT-base parameter set as 19652 as (Silver et al., 2017) suggests.
        """
        self.c1 = c1
        self.c2 = c2
        self.discount = discount

    def puct_score(self, node: Node) -> float:
        parent = cast(Node, node.parent)

        pb_c = sqrt(parent.visit_count) / (node.visit_count + 1)
        pb_c *= log((node.visit_count + self.c2 + 1) / self.c2) + self.c1

        prior_score = pb_c * node.policy_priority
        if node.visit_count > 0:
            ts = node.value_sum / node.visit_count
        else:
            ts = 0
        return ts + prior_score

    def __call__(self, root: Node) -> Node:
        while len(root.children.values()) > 0:
            root = max(root.children.values(), key=self.puct_score)
        return root
