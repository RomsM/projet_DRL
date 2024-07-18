import numpy as np

class LineWorld:
    def __init__(self, length, start, goal, step_reward=-0.01, goal_reward=1.0):
        self.length = length
        self.start = start
        self.goal = goal
        self.state = start
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.action_space = np.array([0, 1])  # 0: left, 1: right
        self.action_space_size = len(self.action_space)  # For compatibility
        self.observation_space = np.arange(length)
        self.observation_space_size = len(self.observation_space)  # For compatibility
        self.P = self._create_transition_probabilities()

    def _create_transition_probabilities(self):
        P = {}
        for s in range(self.length):
            P[s] = {a: [] for a in self.action_space}
            for action in self.action_space:
                next_state, reward, done = self._take_action(s, action)
                prob = 1.0
                P[s][action].append((prob, next_state, reward, done))
        return P

    def _take_action(self, state, action):
        if action == 0:
            next_state = max(0, state - 1)
        elif action == 1:
            next_state = min(self.length - 1, state + 1)

        reward = self.step_reward
        done = False
        if next_state == self.goal:
            reward = self.goal_reward
            done = True

        return next_state, reward, done

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        next_state, reward, done = self._take_action(self.state, action)
        self.state = next_state
        return self.state, reward, done, {}

    def render(self):
        line = ['-'] * self.length
        line[self.start] = 'A'
        line[self.goal] = 'G'
        line[self.state] = 'O'
        print(''.join(line))
