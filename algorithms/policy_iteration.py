import numpy as np

class PolicyIteration:
    def __init__(self, env, gamma=0.99, theta=0.0001):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.policy = np.ones((len(env.observation_space), len(env.action_space))) / len(env.action_space)
        self.value_function = np.zeros(len(env.observation_space))

    def policy_evaluation(self):
        while True:
            delta = 0
            for s in range(len(self.env.observation_space)):
                v = 0
                for a, action_prob in enumerate(self.policy[s]):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        v += action_prob * prob * (reward + self.gamma * self.value_function[next_state])
                delta = max(delta, np.abs(v - self.value_function[s]))
                self.value_function[s] = v
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for s in range(len(self.env.observation_space)):
            chosen_a = np.argmax(self.policy[s])
            action_values = np.zeros(len(self.env.action_space))
            for a in range(len(self.env.action_space)):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    action_values[a] += prob * (reward + self.gamma * self.value_function[next_state])
            best_a = np.argmax(action_values)
            if chosen_a != best_a:
                policy_stable = False
            self.policy[s] = np.eye(len(self.env.action_space))[best_a]
        return policy_stable

    def train(self):
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break

    def get_policy(self):
        return self.policy

    def get_value_function(self):
        return self.value_function

# Test
def test_policy_iteration(env):
    agent = PolicyIteration(env)
    agent.train()
    policy = agent.get_policy()
    value_function = agent.get_value_function()

    print("Politique obtenue:")
    print(policy)
    print("Fonction de valeur obtenue:")
    print(value_function)

    actions = np.argmax(policy, axis=1)
    print("Actions suggérées par la politique pour chaque état:")
    print(actions)

    total_reward = 0
    state = env.reset()
    for _ in range(100):
        env.render()
        action = np.argmax(policy[state])
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    env.render()
    print("Total des récompenses reçues en suivant la politique:", total_reward)

if __name__ == "__main__":
    from environments.grid_world import GridWorld
    from environments.line_world import LineWorld

    env = LineWorld(length=10, start=0, goal=9)
    test_policy_iteration(env)

    env = GridWorld(5, 5, (0, 0), (4, 4), [(1, 1), (2, 2), (3, 3)])
    test_policy_iteration(env)
