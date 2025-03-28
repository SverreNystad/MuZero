from src.search.nodes import Node
from src.search.strategies import BackpropagationStrategy


class Backpropagation(BackpropagationStrategy):
    def __init__(self, discount: float) -> None:
        self.discount = discount

    def __call__(self, leaf_node: Node, rewards: list[float], to_play: int) -> None:
        node: Node | None = leaf_node
        discounted_return = 0.0
        for reward in reversed(rewards):
            discounted_return = reward + self.discount * discounted_return

        while node is not None:
            node.visit_count += 1
            node.value_sum += discounted_return if node.to_play == to_play else -discounted_return
            node = node.parent
