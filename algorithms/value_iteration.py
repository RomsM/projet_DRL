import numpy as np
import logging

class ValueIteration:
    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(env.observation_space.n)
        self.policy = np.zeros(env.observation_space.n, dtype=int)
    
    def calculate_value(self, state, action):
        """Calcule la valeur attendue pour un état et une action donnés."""
        total = 0
        for prob, next_state, reward, done in self.env.P[state][action]:
            total += prob * (reward + self.gamma * self.V[next_state])
        return total
    
    def policy_evaluation(self):
        while True:
            delta = 0
            for state in range(self.env.observation_space.n):
                v = self.V[state]
                self.V[state] = max([self.calculate_value(state, action) for action in range(self.env.action_space.n)])
                delta = max(delta, abs(v - self.V[state]))
            if delta < self.theta:
                break
    
    def policy_improvement(self):
        policy_stable = True
        for state in range(self.env.observation_space.n):
            old_action = self.policy[state]
            action_values = np.zeros(self.env.action_space.n)
            for action in range(self.env.action_space.n):
                action_values[action] = self.calculate_value(state, action)
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
            logging.info(f"Iteration: {iteration}")
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
