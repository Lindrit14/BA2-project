# dynamic_maze.py
import numpy as np
import random

class DynamicMaze:
    def __init__(self, rows=9, cols=9, start=(0,0), goal=(8,8)):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal

        # 0=free, 1=wall
        self.grid = np.zeros((rows, cols))
        # place random walls
        for _ in range(6):
            r = np.random.randint(rows)
            c = np.random.randint(cols)
            self.grid[r,c] = 1
        self.grid[self.start] = 0  # ensure start is free
        self.grid[self.goal] = 0   # ensure goal is free
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        # same logic as static version
        next_pos = list(self.agent_pos)
        if action == 0:
            next_pos[0] -= 1
        elif action == 1:
            next_pos[0] += 1
        elif action == 2:
            next_pos[1] -= 1
        elif action == 3:
            next_pos[1] += 1

        r, c = next_pos
        if r<0 or r>=self.rows or c<0 or c>=self.cols or self.grid[r,c] == 1:
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

        # Now do some dynamic changes each step
        self.make_maze_shift()

        return self.agent_pos, reward, done

    def make_maze_shift(self):
        # randomly flip a few cells
        for _ in range(2):  # maybe flip 2 cells each time
            rr = np.random.randint(self.rows)
            cc = np.random.randint(self.cols)
            if (rr,cc) != self.start and (rr,cc) != self.goal:
                if self.grid[rr,cc] == 0:
                    self.grid[rr,cc] = 1
                else:
                    self.grid[rr,cc] = 0
