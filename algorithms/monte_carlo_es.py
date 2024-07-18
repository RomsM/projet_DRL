import numpy as np

def monte_carlo_es(env, num_episodes=1000, gamma=1.0):
    policy = np.zeros(env.length)
    value_function = np.zeros(env.length)
    returns = {s: [] for s in range(env.length)}

    for episode in range(num_episodes):
        episode_data = []
        state = env.reset()
        done = False
        while not done:
            action = np.random.choice(env.action_space)
            next_state, reward, done, _ = env.step(action)
            episode_data.append((state, action, reward))
            state = next_state
        
        G = 0
        for state, action, reward in reversed(episode_data):
            G = gamma * G + reward
            if not any(s == state for s, _, _ in episode_data[:-1]):
                returns[state].append(G)
                value_function[state] = np.mean(returns[state])
                policy[state] = np.argmax([sum([p * (r + gamma * value_function[s_]) for p, s_, r, _ in env.P[state][a]]) for a in env.action_space])

    return policy, value_function
