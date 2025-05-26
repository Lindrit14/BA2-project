# environments/static_maze.py
import numpy as np
import random

class StaticMaze:
    def __init__(self, rows=7, cols=7, start=(0, 0), goal=(6, 6), wall_count=10):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.wall_count = wall_count
        self.grid = np.zeros((rows, cols))
        self._generate_maze()
        self.reset()

    def _generate_maze(self):
        for _ in range(self.wall_count):
            r, c = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if (r, c) not in [self.start, self.goal]:
                self.grid[r, c] = 1

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        next_pos = list(self.agent_pos)
        if action == 0: next_pos[0] -= 1  # up
        elif action == 1: next_pos[0] += 1  # down
        elif action == 2: next_pos[1] -= 1  # left
        elif action == 3: next_pos[1] += 1  # right

        r, c = next_pos
        if 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 0:
            self.agent_pos = (r, c)

        if self.agent_pos == self.goal:
            return self.agent_pos, 100, True  # reward, done
        return self.agent_pos, -0.1, False
