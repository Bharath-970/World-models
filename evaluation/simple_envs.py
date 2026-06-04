"""
Lightweight MiniGrid Empty-5x5 simulator for fast CEM rollouts.

Avoids the env cloning problem by re-implementing the simple Empty-5x5
dynamics: a 5x5 grid, agent at (1,1), goal at (3,3), 4-direction movement
plus the 4 unused actions. Sparse reward: 1 on terminal when agent at goal.

Used to make MPC tractable on CPU.
"""
import numpy as np
import gymnasium as gym
import minigrid


class EmptyGrid:
    """Simple MiniGrid-Empty-5x5 clone."""

    WIDTH = 5
    HEIGHT = 5
    AGENT_START = (1, 1)
    GOAL = (3, 3)

    def __init__(self):
        self.agent_pos = list(self.AGENT_START)
        self.agent_dir = 0  # 0=right, 1=down, 2=left, 3=up
        self.done = False
        self.step_count = 0
        self.max_steps = 256
        self.goal_reached = False

    def clone(self):
        g = EmptyGrid()
        g.agent_pos = list(self.agent_pos)
        g.agent_dir = self.agent_dir
        g.done = self.done
        g.step_count = self.step_count
        g.max_steps = self.max_steps
        g.goal_reached = self.goal_reached
        return g

    def step(self, action):
        if self.done:
            return 0.0, True, False

        if action == 0:  # turn left
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == 1:  # turn right
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 2:  # forward
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][self.agent_dir]
            nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                self.agent_pos = [nx, ny]
        # actions 3-6 (pickup, drop, toggle, done) are no-ops in Empty

        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        if tuple(self.agent_pos) == self.GOAL:
            reward = 1.0
            self.goal_reached = True
            terminated = True
        if self.step_count >= self.max_steps:
            truncated = True
        if terminated or truncated:
            self.done = True

        return reward, terminated, truncated


class FourRoomsGrid:
    """MiniGrid-FourRooms clone with 4 rooms separated by walls + 4 doorways."""

    WIDTH = 19
    HEIGHT = 19
    WALLS = {
        # horizontal walls in top half
        *[(x, 6) for x in [3, 4, 5, 8, 9, 10, 13, 14, 15]],
        # horizontal walls in bottom half
        *[(x, 12) for x in [3, 4, 5, 8, 9, 10, 13, 14, 15]],
        # vertical walls in left half
        *[(6, y) for y in [3, 4, 5, 8, 9, 10, 13, 14, 15]],
        # vertical walls in right half
        *[(12, y) for y in [3, 4, 5, 8, 9, 10, 13, 14, 15]],
    }
    AGENT_START = (1, 1)
    GOAL_OPTIONS = [(17, 17), (17, 1), (1, 17), (1, 1)]

    def __init__(self, max_steps=512):
        self.max_steps = max_steps
        self.goal = self.GOAL_OPTIONS[3]  # default
        self.reset()

    def reset(self):
        self.agent_pos = list(self.AGENT_START)
        self.agent_dir = 0
        self.step_count = 0
        self.done = False
        return

    def clone(self):
        g = FourRoomsGrid(max_steps=self.max_steps)
        g.agent_pos = list(self.agent_pos)
        g.agent_dir = self.agent_dir
        g.step_count = self.step_count
        g.done = self.done
        g.goal = self.goal
        return g

    def step(self, action):
        if self.done:
            return 0.0, True, False

        if action == 0:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == 1:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 2:
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][self.agent_dir]
            nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT and (nx, ny) not in self.WALLS:
                self.agent_pos = [nx, ny]
        # actions 3-6 no-op

        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False

        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            terminated = True
        if self.step_count >= self.max_steps:
            truncated = True
        if terminated or truncated:
            self.done = True

        return reward, terminated, truncated
