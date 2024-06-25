import numpy as np
from gym.spaces import Discrete

class LineWorld:
    def __init__(self, length, start, goal):
        self.length = length
        self.start = start
        self.goal = goal
        self.state = start
        self.observation_space = Discrete(length)
        self.action_space = Discrete(2)
        self.P = self._build_transition_matrix()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        if action == 0:  # Move left
            next_state = max(0, self.state - 1)
        elif action == 1:  # Move right
            next_state = min(self.length - 1, self.state + 1)
        else:
            raise ValueError("Invalid action")

        reward = -1 if next_state != self.goal else 0
        done = next_state == self.goal
        self.state = next_state
        return next_state, reward, done, {}

    def _build_transition_matrix(self):
        P = {}
        for state in range(self.length):
            P[state] = {0: [], 1: []}
            if state > 0:
                P[state][0].append((1.0, state - 1, -1, state - 1 == self.goal))
            else:
                P[state][0].append((1.0, state, -1, state == self.goal))
            if state < self.length - 1:
                P[state][1].append((1.0, state + 1, -1, state + 1 == self.goal))
            else:
                P[state][1].append((1.0, state, -1, state == self.goal))
        return P

    def render(self):
        print(f"State: {self.state}")
