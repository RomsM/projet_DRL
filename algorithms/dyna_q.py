import numpy as np
from collections import defaultdict

class DynaQ:
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.1, planning_steps=10):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        self.Q = np.zeros((env.observation_space_size, env.action_space_size))
        self.model = defaultdict(lambda: defaultdict(lambda: (0, 0, False)))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space)
        else:
            return np.argmax(self.Q[state])

    def update_model(self, state, action, reward, next_state, done):
        self.model[state][action] = (reward, next_state, done)

    def planning_step(self):
        for _ in range(self.planning_steps):
            state = np.random.choice(list(self.model.keys()))
            action = np.random.choice(list(self.model[state].keys()))
            reward, next_state, done = self.model[state][action]
            td_target = reward + self.gamma * np.max(self.Q[next_state]) * (not done)
            td_error = td_target - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error

    def train(self, num_episodes=1000):
        for i in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                td_target = reward + self.gamma * np.max(self.Q[next_state]) * (not done)
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error
                self.update_model(state, action, reward, next_state, done)
                self.planning_step()
                state = next_state

    def get_policy(self):
        return np.argmax(self.Q, axis=1)

    def get_action_value_function(self):
        return self.Q
