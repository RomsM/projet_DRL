import numpy as np
from collections import defaultdict

class OffPolicyMCC:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.C = defaultdict(lambda: np.zeros(env.action_space.n))
        self.target_policy = np.zeros([env.observation_space.n, env.action_space.n])
        self.behavior_policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        while True:
            action = np.random.choice(self.env.action_space.n, p=self.behavior_policy[state])
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode

    def train(self, num_episodes=1000):
        for _ in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            W = 1
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                self.C[state][action] += W
                self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])
                best_action = np.argmax(self.Q[state])
                self.target_policy[state] = np.eye(self.env.action_space.n)[best_action]
                if action != best_action:
                    break
                W = W / self.behavior_policy[state][action]

    def get_policy(self):
        return self.target_policy

    def get_action_value_function(self):
        return self.Q
