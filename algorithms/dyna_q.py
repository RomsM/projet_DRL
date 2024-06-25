import numpy as np
from collections import defaultdict

class DynaQ:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, n_planning_steps=5):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning_steps = n_planning_steps
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        self.model = defaultdict(lambda: defaultdict(lambda: (0, 0, 0)))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            while True:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                best_next_action = np.argmax(self.Q[next_state])
                td_target = reward + self.gamma * self.Q[next_state, best_next_action]
                td_error = td_target - self.Q[state, action]
                self.Q[state, action] += self.alpha * td_error
                self.model[state][action] = (reward, next_state, done)
                self.planning()
                state = next_state
                if done:
                    break
            print(f"Episode {episode + 1}/{num_episodes} terminé")

    def planning(self):
        for _ in range(self.n_planning_steps):
            state = np.random.choice(list(self.model.keys()))
            action = np.random.choice(list(self.model[state].keys()))
            reward, next_state, done = self.model[state][action]
            best_next_action = np.argmax(self.Q[next_state])
            td_target = reward + self.gamma * self.Q[next_state, best_next_action]
            td_error = td_target - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error

    def get_policy(self):
        policy = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        for s in range(self.env.observation_space.n):
            best_action = np.argmax(self.Q[s])
            policy[s] = np.eye(self.env.action_space.n)[best_action]
        return policy

    def get_action_value_function(self):
        return self.Q
