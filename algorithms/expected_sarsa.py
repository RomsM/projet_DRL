import numpy as np

class ExpectedSarsa:
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.zeros((env.observation_space_size, env.action_space_size))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space)
        else:
            return np.argmax(self.Q[state])

    def train(self, num_episodes=1000):
        for i in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                expected_value = np.dot(
                    [self.epsilon / len(self.env.action_space) + (1 - self.epsilon) if a == np.argmax(self.Q[next_state]) else self.epsilon / len(self.env.action_space)
                     for a in self.env.action_space],
                    self.Q[next_state]
                )
                td_target = reward + self.gamma * expected_value
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error
                state = next_state

    def get_policy(self):
        return np.argmax(self.Q, axis=1)

    def get_action_value_function(self):
        return self.Q
