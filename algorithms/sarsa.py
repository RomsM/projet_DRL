import numpy as np

class Sarsa:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def train(self, num_episodes):
        for _ in range(num_episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            done = False
            while not done:
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.choose_action(next_state)
                self.q_table[state, action] += self.alpha * (
                    reward + self.gamma * self.q_table[next_state, next_action] - self.q_table[state, action]
                )
                state, action = next_state, next_action

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)

    def get_action_value_function(self):
        return self.q_table

    def save(self, filename):
        np.savez(filename, q_table=self.q_table)

    def load(self, filename):
        data = np.load(filename)
        self.q_table = data['q_table']
