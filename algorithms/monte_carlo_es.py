import numpy as np
from collections import defaultdict

class MonteCarloES:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
        self.returns_count = defaultdict(lambda: np.zeros(env.action_space.n))
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        steps = 0
        max_steps = 100  # Limite le nombre de pas par épisode pour éviter les boucles infinies
        while True:
            action = np.random.choice(self.env.action_space.n, p=self.policy[state])
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            steps += 1
            if done or steps >= max_steps:
                break
            state = next_state
        print(f"Episode generated with {steps} steps.")
        return episode

    def train(self, num_episodes=10):  # Réduire le nombre d'épisodes pour les tests
        for i in range(num_episodes):
            episode = self.generate_episode()
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                if not any((x[0] == state and x[1] == action) for x in episode[:t]):
                    self.returns_sum[state][action] += G
                    self.returns_count[state][action] += 1
                    self.Q[state][action] = self.returns_sum[state][action] / self.returns_count[state][action]
                    best_action = np.argmax(self.Q[state])
                    self.policy[state] = np.eye(self.env.action_space.n)[best_action]
            print(f"Episode {i + 1}/{num_episodes} terminé")

    def get_policy(self):
        return self.policy

    def get_action_value_function(self):
        return self.Q
