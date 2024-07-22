import numpy as np
import logging


# Utilise un seuil (theta) pour déterminer la convergence.

class PolicyIteration:
    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.policy = np.zeros(env.observation_space.n, dtype=int)
        self.V = np.zeros(env.observation_space.n)

    def policy_evaluation(self):
        while True:
            delta = 0
            for state in range(self.env.observation_space.n):
                v = self.V[state]
                action = self.policy[state]
                self.V[state] = sum([prob * (reward + self.gamma * self.V[next_state])
                                     for prob, next_state, reward, done in self.env.P[state][action]])
                delta = max(delta, abs(v - self.V[state]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for state in range(self.env.observation_space.n):
            old_action = self.policy[state]
            action_values = np.zeros(self.env.action_space.n)
            for action in range(self.env.action_space.n):
                action_values[action] = sum([prob * (reward + self.gamma * self.V[next_state])
                                             for prob, next_state, reward, done in self.env.P[state][action]])
            new_action = np.argmax(action_values)
            self.policy[state] = new_action
            if old_action != new_action:
                policy_stable = False
        return policy_stable

    def train(self):
        logging.info("Starting training...")
        iteration = 0
        while True:
            iteration += 1
            logging.info(f"Policy Iteration {iteration}")
            self.policy_evaluation()
            if self.policy_improvement():
                break

    def get_policy(self):
        return self.policy

    def get_value_function(self):
        return self.V

    def save(self, filepath):
        np.savez(filepath, policy=self.policy, value_function=self.V)

    def load(self, filepath):
        data = np.load(filepath)
        self.policy = data['policy']
        self.V = data['value_function']
