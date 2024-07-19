import numpy as np

class OffPolicyMCC:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.c_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.policy = np.ones((env.observation_space.n, env.action_space.n)) / env.action_space.n

    def generate_episode(self, policy):
        episode = []
        state = self.env.reset()
        while True:
            action = np.random.choice(np.arange(self.env.action_space.n), p=policy[state])
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode

    def train(self, num_episodes=1000):
        behavior_policy = np.ones((self.env.observation_space.n, self.env.action_space.n)) / self.env.action_space.n
        for i in range(num_episodes):
            episode = self.generate_episode(behavior_policy)
            g = 0.0
            w = 1.0
            for state, action, reward in reversed(episode):
                g = self.gamma * g + reward
                self.c_table[state, action] += w
                self.q_table[state, action] += (w / self.c_table[state, action]) * (g - self.q_table[state, action])
                self.policy[state] = np.argmax(self.q_table[state])
                if action != np.argmax(self.q_table[state]):
                    break
                w /= behavior_policy[state, action]

    def get_policy(self):
        return np.argmax(self.q_table, axis=1)

    def get_action_value_function(self):
        return self.q_table

    def save(self, filename):
        np.savez(filename, q_table=self.q_table, policy=self.policy)

    def load(self, filename):
        data = np.load(filename)
        self.q_table = data['q_table']
        self.policy = data['policy']
