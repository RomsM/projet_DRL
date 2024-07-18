import sys
import os

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend non-graphique de Matplotlib
import matplotlib.pyplot as plt
from environments.line_world import LineWorld
from algorithms.monte_carlo_es import MonteCarloES
from algorithms.off_policy_mcc import OffPolicyMCC
from algorithms.on_policy_first_visit_mcc import OnPolicyFirstVisitMCC
from algorithms.sarsa import SARSA
from algorithms.expected_sarsa import ExpectedSARSA
from algorithms.q_learning import QLearning
from algorithms.dyna_q import DynaQ

def evaluate_algorithm(env, alg_class, num_episodes=1000, max_steps=100):
    agent = alg_class(env)
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            if hasattr(agent, 'update'):
                agent.update(state, action, reward, next_state, done)
            elif hasattr(agent, 'learn'):
                agent.learn(state, action, reward, next_state, done)
            if hasattr(agent, 'planning_step') and hasattr(agent, 'model') and len(agent.model) > 0:
                agent.planning_step()
            state = next_state
            total_reward += reward
            if done:
                break
        else:
            print(f"Episode {episode} reached max steps without termination.")

        total_rewards.append(total_reward)

    return total_rewards

if __name__ == "__main__":
    env = LineWorld(length=10, start=0, goal=9)
    algorithms = [
        MonteCarloES,
        OffPolicyMCC,
        OnPolicyFirstVisitMCC,
        SARSA,
        ExpectedSARSA,
        QLearning,
        DynaQ,
    ]

    for alg_class in algorithms:
        print(f"Evaluating {alg_class.__name__}...")
        rewards = evaluate_algorithm(env, alg_class)
        print(f"Average reward for {alg_class.__name__}: {np.mean(rewards)}")
        plt.plot(rewards, label=alg_class.__name__)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Algorithm Comparison')
    plt.legend()
    plt.savefig('algorithm_comparison.png')  # Sauvegarder le graphique au lieu de l'afficher
