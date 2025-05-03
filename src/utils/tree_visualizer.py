from typing import Any, Tuple
import time
import pygame
from src.search.nodes import Node
import math

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

class TreeVisualizer:
    def __init__(self, root: Node) -> None:
        self.screen: Any = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        pygame.display.set_caption("Tree Visualizer")

        self.root = root
        self.radius = 5
        self.y_step = 80
        self.base_x = 40
        self.pan_speed = 40
        self.zoom = 1.0
        self.offset = (WINDOW_WIDTH // 2, 20)

        self.root.pos = (0, 0)

    def __call__(self, tick_callable, max_itr) -> None:
        pygame.init()
        self.running = True
        clock = pygame.time.Clock()
        itr = 0

        while itr < max_itr:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        self.running = False
                    elif ev.key == pygame.K_LEFT:
                        self.offset = (self.offset[0] + self.pan_speed,
                                       self.offset[1])
                    elif ev.key == pygame.K_RIGHT:
                        self.offset = (self.offset[0] - self.pan_speed,
                                       self.offset[1])
                    elif ev.key == pygame.K_UP:
                        self.offset = (self.offset[0],
                                       self.offset[1] + self.pan_speed)
                    elif ev.key == pygame.K_DOWN:
                        self.offset = (self.offset[0],
                                       self.offset[1] - self.pan_speed)

            self.screen.fill((255, 255, 255))
            self.bfs_draw()
            pygame.display.flip()

            if tick_callable:
                tick_callable()
                itr += 1
                time.sleep(1)
            clock.tick(60)

        pygame.quit()

    def to_screen(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Map world coords → screen pixels."""
        x = world_pos[0] * self.zoom + self.offset[0]
        y = world_pos[1] * self.zoom + self.offset[1]
        return (int(x), int(y))

    def _draw_node(self, screen_pos: Tuple[int, int]) -> None:
        pygame.draw.circle(self.screen, (255, 0, 0), screen_pos, self.radius)

    def _get_depth(self, node: Node) -> int:
        """Get the depth of the node in the tree."""
        depth = 0
        while node.children:
            node = list(node.children.values())[0]
            depth += 1
        return depth

    def _check_if_any_nodes_collide(self, pos: Tuple[int , int]) -> bool:
        """Check if any nodes collide with each other."""
        queue = [self.root]

        while len(queue) > 0:
            node = queue.pop(0)
            if node.pos == pos:
                return True

            for child in node.children.values():
                queue.append(child)

        return False

    def bfs_draw(self) -> None:
        queue = [(self.root, 0)]

        while queue:
            node, depth = queue.pop(0)
            p_screen = self.to_screen(node.pos)
            self._draw_node(p_screen)

            for i, child in enumerate(node.children.values()):
                sign = -1 if i == 0 else 1

                dx = sign * self.base_x * (self._get_depth(node)/(abs(self._get_depth(node) - self._get_depth(self.root)) + 1) * self._get_depth(self.root) / 4)

                pos = (node.pos[0] + dx,
                    node.pos[1] + self.y_step)
                collision = self._check_if_any_nodes_collide(pos)

                if collision:
                    pos = (pos[0] + self.radius * 3, pos[1])

                child.pos = pos
                c_screen = self.to_screen(child.pos)
                pygame.draw.line(self.screen,
                                 (0, 0, 0),
                                 p_screen,
                                 c_screen,
                                 1)

                queue.append((child, depth + 1))
