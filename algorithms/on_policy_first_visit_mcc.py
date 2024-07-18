import numpy as np

class OnPolicyFirstVisitMCC:
    def __init__(self, env, gamma=1.0):
        self.env = env
        self.gamma = gamma
        self.Q = np.zeros((env.observation_space_size, env.action_space_size))
        self.returns = [[[] for _ in range(env.action_space_size)] for _ in range(env.observation_space_size)]
        self.policy = np.zeros(env.observation_space_size, dtype=int)

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        done = False
        while not done:
            action = np.random.choice(self.env.action_space)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def train(self, num_episodes=1000):
        for i in range(num_episodes):
            episode = self.generate_episode()
            states, actions, rewards = zip(*episode)
            G = 0
            for t in range(len(states) - 1, -1, -1):
                G = self.gamma * G + rewards[t]
                if (states[t], actions[t]) not in list(zip(states[:t], actions[:t])):
                    self.returns[states[t]][actions[t]].append(G)
                    self.Q[states[t], actions[t]] = np.mean(self.returns[states[t]][actions[t]])
                    self.policy[states[t]] = np.argmax(self.Q[states[t]])

    def get_policy(self):
        return self.policy

    def get_action_value_function(self):
        return self.Q
