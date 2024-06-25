import numpy as np

class ValueIteration:
    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.value_function = np.zeros(env.observation_space.n)

    def train(self):
        while True:
            delta = 0
            for s in range(self.env.observation_space.n):
                v = self.value_function[s]
                action_values = np.zeros(self.env.action_space.n)
                for a in range(self.env.action_space.n):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        action_values[a] += prob * (reward + self.gamma * self.value_function[next_state])
                self.value_function[s] = np.max(action_values)
                delta = max(delta, np.abs(v - self.value_function[s]))
            if delta < self.theta:
                break

    def get_policy(self):
        policy = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        for s in range(self.env.observation_space.n):
            action_values = np.zeros(self.env.action_space.n)
            for a in range(self.env.action_space.n):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    action_values[a] += prob * (reward + self.gamma * self.value_function[next_state])
            best_action = np.argmax(action_values)
            policy[s] = np.eye(self.env.action_space.n)[best_action]
        return policy

    def get_value_function(self):
        return self.value_function
