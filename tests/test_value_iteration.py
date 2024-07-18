import sys
import os
import numpy as np

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.grid_world import GridWorld
from environments.line_world import LineWorld
from algorithms.value_iteration import ValueIteration

def test_value_iteration(env):
    agent = ValueIteration(env)
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
    for step in range(100):
        print(f"Step {step}: State={state}")
        env.render()
        action = np.argmax(policy[state])
        print(f"Action choisie: {action}")
        state, reward, done, _ = env.step(action)
        print(f"Nouvel état: {state}, Récompense: {reward}, Terminé: {done}")
        total_reward += reward
        if done:
            print("État terminal atteint.")
            break
    env.render()
    print("Total des récompenses reçues en suivant la politique:", total_reward)

if __name__ == "__main__":
    env = LineWorld(length=10, start=0, goal=9)
    test_value_iteration(env)

    env = GridWorld(5, 5, (0, 0), (4, 4), [(1, 1), (2, 2), (3, 3)])
    test_value_iteration(env)
