import numpy as np

class Sarsa:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def train(self, num_episodes):
        for _ in range(num_episodes):
            state = self.env.reset()
            action = self.select_action(state)
            done = False
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.select_action(next_state)
                self.update_q_table(state, action, reward, next_state, next_action)
                state, action = next_state, next_action

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, next_action):
        td_target = reward + self.gamma * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)

    def get_action_value_function(self):
        return self.q_table

    def save(self, filename):
        np.save(filename, self.q_table)

    def load(self, filename):
        self.q_table = np.load(filename)
