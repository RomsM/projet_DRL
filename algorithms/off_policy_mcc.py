import numpy as np

class OffPolicyMCC:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma
        self.Q = np.zeros((env.length, len(env.action_space)))
        self.C = np.zeros((env.length, len(env.action_space)))
        self.policy = np.zeros(env.length, dtype=int)

    def train(self, num_episodes):
        for _ in range(num_episodes):
            behavior_policy = lambda state: np.random.choice(self.env.action_space)
            episode = self.generate_episode(behavior_policy)
            G = 0
            W = 1
            for state, action, reward in reversed(episode):
                G = self.gamma * G + reward
                self.C[state, action] += W
                self.Q[state, action] += (W / self.C[state, action]) * (G - self.Q[state, action])
                self.policy[state] = np.argmax(self.Q[state])
                if action != self.policy[state]:
                    break
                W /= 1.0 / len(self.env.action_space)

    def generate_episode(self, policy):
        episode = []
        state = self.env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def get_policy(self):
        return self.policy

    def get_action_value_function(self):
        return self.Q
