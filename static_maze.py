# static_maze.py
import numpy as np

class StaticMaze:
    def __init__(self, rows=9, cols=9, start=(0,0), goal=(8,8)):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal

        # Example: 0 means free cell, 1 means wall
        self.grid = np.zeros((rows, cols))
        # You can add some walls
        self.grid[1,2] = 1
        self.grid[2,2] = 1
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos  # Return initial state

    def step(self, action):
        # action: 0=up, 1=down, 2=left, 3=right
        next_pos = list(self.agent_pos)
        if action == 0:   # up
            next_pos[0] -= 1
        elif action == 1: # down
            next_pos[0] += 1
        elif action == 2: # left
            next_pos[1] -= 1
        elif action == 3: # right
            next_pos[1] += 1

        # Check boundaries and walls
        r, c = next_pos
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols or self.grid[r,c] == 1:
            # invalid move => no change in position, negative reward maybe
            reward = -1
            done = False
            next_pos = self.agent_pos
        else:
            self.agent_pos = (r, c)
            if self.agent_pos == self.goal:
                reward = 100
                done = True
            else:
                reward = -0.1
                done = False

        return self.agent_pos, reward, done
