import numpy as np

class PolicyIteration:
    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.policy = np.zeros([env.observation_space.n, env.action_space.n])
        self.value_function = np.zeros(env.observation_space.n)

    def policy_evaluation(self):
        while True:
            delta = 0
            for s in range(self.env.observation_space.n):
                v = self.value_function[s]
                new_value = 0
                for a in range(self.env.action_space.n):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        new_value += self.policy[s, a] * prob * (reward + self.gamma * self.value_function[next_state])
                self.value_function[s] = new_value
                delta = max(delta, np.abs(v - self.value_function[s]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for s in range(self.env.observation_space.n):
            old_action = np.argmax(self.policy[s])
            action_values = np.zeros(self.env.action_space.n)
            for a in range(self.env.action_space.n):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    action_values[a] += prob * (reward + self.gamma * self.value_function[next_state])
            best_action = np.argmax(action_values)
            self.policy[s] = np.eye(self.env.action_space.n)[best_action]
            if old_action != best_action:
                policy_stable = False
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
