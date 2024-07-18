import numpy as np

class ValueIteration:
    def __init__(self, env, gamma=0.99, theta=0.0001):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.value_function = np.zeros(len(env.observation_space))
        self.policy = np.zeros((len(env.observation_space), len(env.action_space)))

    def train(self):
        while True:
            delta = 0
            for s in range(len(self.env.observation_space)):
                v = self.value_function[s]
                action_values = np.zeros(len(self.env.action_space))
                for a in range(len(self.env.action_space)):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        action_values[a] += prob * (reward + self.gamma * self.value_function[next_state])
                self.value_function[s] = np.max(action_values)
                delta = max(delta, np.abs(v - self.value_function[s]))
            if delta < self.theta:
                break
        self._extract_policy()

    def _extract_policy(self):
        for s in range(len(self.env.observation_space)):
            action_values = np.zeros(len(self.env.action_space))
            for a in range(len(self.env.action_space)):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    action_values[a] += prob * (reward + self.gamma * self.value_function[next_state])
            best_action = np.argmax(action_values)
            self.policy[s] = np.eye(len(self.env.action_space))[best_action]

    def get_policy(self):
        return self.policy

    def get_value_function(self):
        return self.value_function
