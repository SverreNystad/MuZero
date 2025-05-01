from typing import Any, Tuple
import time
import pygame
from src.search.nodes import Node

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

class TreeVisualizer:
    def __init__(self, root: Node) -> None:
        self.screen: Any = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        pygame.display.set_caption("Tree Visualizer")

        self.root = root

        # visual parameters
        self.radius = 5
        self.y_step = 80          # fixed vertical step
        self.base_x = 40          # base horizontal step
        self.pan_speed = 40       # pixels to move per arrow-key press

        # world-to-screen transform
        # start centered (root at world (0,0) → screen (center,20))
        self.zoom = 1.0
        self.offset = (WINDOW_WIDTH // 2, 20)

        # initialize root world-coords
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
                time.sleep(0.01)

            clock.tick(60)  # up to 30 FPS

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

    def bfs_draw(self) -> None:
        # BFS queue holds (node, depth)
        queue = [(self.root, 0)]

        while queue:
            node, depth = queue.pop(0)
            p_screen = self.to_screen(node.pos)

            # draw the node
            self._draw_node(p_screen)

            # draw & position its children
            for i, child in enumerate(node.children.values()):
                # more spacing the deeper you go: base_x * (depth+1)
                sign = -1 if i == 0 else 1

                dx = sign * self.base_x * (self._get_depth(node))

                child.pos = (node.pos[0] + dx, node.pos[1] + self.y_step)
                c_screen = self.to_screen(child.pos)

                # draw line parent → child
                pygame.draw.line(self.screen,
                                 (0, 0, 0),
                                 p_screen,
                                 c_screen,
                                 1)

                queue.append((child, depth + 1))
