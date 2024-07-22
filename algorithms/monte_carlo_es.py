import numpy as np
import random


class MonteCarloES:
    def __init__(self, env, gamma=0.99, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.returns = {(s, a): [] for s in range(env.observation_space.n) for a in range(env.action_space.n)}
        self.policy = np.zeros(env.observation_space.n, dtype=int)

    # Creating an episode based on the actual policy
    def generate_episode(self):
        episode = []
        state = self.env.reset()
        done = False
        step_counter = 0  # Compteur pour limiter le nombre d'étapes
        max_steps = 100  # Limite maximale des étapes pour éviter les boucles infinies

        while not done and step_counter < max_steps:
            if random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample()  # Exploration aléatoire
            else:
                action = self.policy[state]  # Exploitation de la politique actuelle

            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            step_counter += 1
            print(f"Step {step_counter}: state={state}, action={action}, reward={reward}, done={done}")

        if step_counter >= max_steps:
            print("Maximum steps reached in generate_episode")

        return episode

    # MAJ of policy and valu funct
    def train(self, num_episodes):
        for episode_num in range(num_episodes):
            print(f"Generating episode {episode_num + 1}/{num_episodes}...")
            episode = self.generate_episode()
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                if not any((s == state and a == action) for (s, a, r) in episode[:t]):
                    self.returns[(state, action)].append(G)
                    self.q_table[state][action] = np.mean(self.returns[(state, action)])
                    self.policy[state] = np.argmax(self.q_table[state])
            print(f"Episode {episode_num + 1}/{num_episodes} completed.")

    def get_policy(self):
        return self.policy

    def get_action_value_function(self):
        return self.q_table

    def save(self, filename):
        np.savez(filename, q_table=self.q_table, policy=self.policy)

    def load(self, filename):
        data = np.load(filename)
        self.q_table = data['q_table']
        self.policy = data['policy']
