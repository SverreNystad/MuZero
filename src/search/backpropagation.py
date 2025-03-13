from typing import Union
from src.search.nodes import Node
from src.search.strategies import BackpropagationStrategy


class Backpropagation(BackpropagationStrategy):
    def __call__(self, leaf_node: Node, value: float, to_play: int) -> None:
        node: Union[Node, None] = leaf_node
        while node is not None:
            node.visit_count += 1
            node.value_sum += value if node.to_play == to_play else -value
            node = node.parent
