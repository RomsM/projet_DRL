import numpy as np

class PolicyIteration:
    def __init__(self, env, gamma=0.99, theta=0.0001):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
        self.value_function = np.zeros(env.observation_space.n)

    def policy_evaluation(self):
        while True:
            delta = 0
            for s in range(self.env.observation_space.n):
                v = 0
                for a, action_prob in enumerate(self.policy[s]):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        v += action_prob * prob * (reward + self.gamma * self.value_function[next_state])
                delta = max(delta, np.abs(v - self.value_function[s]))
                self.value_function[s] = v
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for s in range(self.env.observation_space.n):
            chosen_a = np.argmax(self.policy[s])
            action_values = np.zeros(self.env.action_space.n)
            for a in range(self.env.action_space.n):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    action_values[a] += prob * (reward + self.gamma * self.value_function[next_state])
            best_a = np.argmax(action_values)
            if chosen_a != best_a:
                policy_stable = False
            self.policy[s] = np.eye(self.env.action_space.n)[best_a]
        return policy_stable

    def train(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break

    def get_policy(self):
        return self.policy

    def get_value_function(self):
        return self.value_function
