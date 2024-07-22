import numpy as np
import random


class DynaQ:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, n_planning_steps=5):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning_steps = n_planning_steps
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.model = {}

    # ε-greedy pour la sélection d'actions
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    # MAJ of Q and the model based on real experience
    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.alpha * (td_target - self.q_table[state][action])
        self.model[(state, action)] = (reward, next_state)

    # MAJ supplém based on simulated experiences
    def planning(self):
        for _ in range(self.n_planning_steps):
            state, action = random.choice(list(self.model.keys()))
            reward, next_state = self.model[(state, action)]
            self.learn(state, action, reward, next_state)

    # Train for num_episodes
    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.learn(state, action, reward, next_state)
                self.planning()
                state = next_state

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)

    def get_action_value_function(self):
        return self.q_table

    def save(self, filename):
        np.savez(filename, q_table=self.q_table)

    def load(self, filename):
        data = np.load(filename)
        self.q_table = data['q_table']
