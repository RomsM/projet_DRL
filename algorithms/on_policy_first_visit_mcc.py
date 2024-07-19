import numpy as np
import logging

class OnPolicyFirstVisitMCC:
    def __init__(self, env, epsilon=0.1, gamma=1.0):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.returns = {}
        self.policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n

    def generate_episode(self):
        logging.info("Generating episode...")
        episode = []
        state = self.env.reset()
        done = False
        while not done:
            action = np.random.choice(self.env.action_space.n, p=self.policy[state])
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if len(episode) > 100:  # pour éviter les boucles infinies
                logging.warning("Episode length exceeded 100 steps, stopping...")
                break
        logging.info("Episode generated.")
        return episode

    def train(self, num_episodes):
        logging.info(f"Starting training for {num_episodes} episodes...")
        for i in range(num_episodes):
            episode = self.generate_episode()
            logging.info(f"Training progress: Episode {i + 1}/{num_episodes}")
            states, actions, rewards = zip(*episode)
            g = 0
            visited = set()
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                g = self.gamma * g + reward
                if (state, action) not in visited:
                    visited.add((state, action))
                    if (state, action) not in self.returns:
                        self.returns[(state, action)] = []
                    self.returns[(state, action)].append(g)
                    self.q_table[state, action] = np.mean(self.returns[(state, action)])
                    best_action = np.argmax(self.q_table[state])
                    self.policy[state] = self.epsilon / self.env.action_space.n
                    self.policy[state, best_action] = 1 - self.epsilon + (self.epsilon / self.env.action_space.n)
        logging.info("Training completed.")

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)

    def get_action_value_function(self):
        return self.q_table

    def save(self, filename):
        np.savez(filename, policy=self.get_policy(), q_table=self.q_table)

    def load(self, filename):
        data = np.load(filename)
        self.policy = data['policy']
        self.q_table = data['q_table']
