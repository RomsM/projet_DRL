import numpy as np
from gym.spaces import Discrete

class GridWorld:
    def __init__(self, width, height, start, goal, obstacles=[]):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.state = start
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.observation_space = Discrete(width * height)
        self.action_space = Discrete(len(self.actions))
        self.P = self._init_transitions()

    def _init_transitions(self):
        P = {}
        for y in range(self.height):
            for x in range(self.width):
                state = self._get_state_index((x, y))
                P[state] = {a: [] for a in range(len(self.actions))}
                if (x, y) not in self.obstacles:
                    for a, action in enumerate(self.actions):
                        next_x, next_y = x, y
                        if action == 'UP':
                            next_y = max(0, y - 1)
                        elif action == 'DOWN':
                            next_y = min(self.height - 1, y + 1)
                        elif action == 'LEFT':
                            next_x = max(0, x - 1)
                        elif action == 'RIGHT':
                            next_x = min(self.width - 1, x + 1)
                        
                        next_state = self._get_state_index((next_x, next_y))
                        if (next_x, next_y) in self.obstacles:
                            next_state = state
                        
                        reward = -1
                        done = False
                        if (next_x, next_y) == self.goal:
                            reward = 0
                            done = True

                        P[state][a].append((1.0, next_state, reward, done))
        return P

    def reset(self):
        self.state = self.start
        return self._get_state_index(self.state)

    def step(self, action):
        x, y = self.state
        if action == 0:  # UP
            y = max(0, y - 1)
        elif action == 1:  # DOWN
            y = min(self.height - 1, y + 1)
        elif action == 2:  # LEFT
            x = max(0, x - 1)
        elif action == 3:  # RIGHT
            x = min(self.width - 1, x + 1)
        
        next_state = (x, y)
        if next_state in self.obstacles:
            next_state = self.state
        
        reward = -1
        done = False
        if next_state == self.goal:
            reward = 0
            done = True
        
        self.state = next_state
        return self._get_state_index(self.state), reward, done, {}

    def render(self):
        grid = np.zeros((self.height, self.width), dtype=int)
        for obs in self.obstacles:
            grid[obs[1], obs[0]] = -1
        grid[self.goal[1], self.goal[0]] = 1
        grid[self.state[1], self.state[0]] = 2
        print(grid)

    def _get_state_index(self, state):
        return state[1] * self.width + state[0]
