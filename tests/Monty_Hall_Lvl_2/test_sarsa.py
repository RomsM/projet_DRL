import sys
import os
import numpy as np

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.monty_hall_level_2 import MontyHallLevel2
from algorithms.sarsa import Sarsa

def test_sarsa():
    print("Initializing environment...")
    env = MontyHallLevel2()
    print("Initializing agent...")
    agent = Sarsa(env)
    print("Starting training...")
    agent.train(num_episodes=100000)  # Augmenté pour assurer une bonne convergence
    policy = agent.get_policy()
    action_value_function = agent.get_action_value_function()

    print("Politique obtenue:")
    print(policy)
    print("Fonction de valeur-action obtenue:")
    print(action_value_function)

    # Sauvegarde de la politique et des fonctions
    agent.save(r'D:\projet_DRL\tests\Monty_hall_Lvl_2\policy\sarsa_policy_MH2.npz')
    print("Politique et fonctions sauvegardées dans 'sarsa_policy_MH2.npz'.")

    # Chargement de la politique et des fonctions
    agent.load(r'D:\projet_DRL\tests\Monty_hall_Lvl_2\policy\sarsa_policy_MH2.npz')
    loaded_policy = agent.get_policy()
    loaded_action_value_function = agent.get_action_value_function()
    print("Politique chargée:")
    print(loaded_policy)
    print("Fonction de valeur-action chargée:")
    print(loaded_action_value_function)

    # Démonstration de la politique
    print("\nDémonstration de la politique:")
    num_test_episodes = 1000
    total_wins = 0
    for _ in range(num_test_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy[state]
            state, reward, done, _ = env.step(action)
        total_wins += reward
    win_rate = total_wins / num_test_episodes
    print(f"Taux de victoire sur {num_test_episodes} épisodes: {win_rate:.2%}")

    # Interaction manuelle
    print("\nInteraction manuelle:")
    state = env.reset()
    print(f"État initial: {state}")
    done = False
    while not done:
        door, step = state
        action = int(input(f"Étape {step + 1}/4: Entrez l'action (0 pour rester, 1 pour changer): "))
        state, reward, done, _ = env.step(action)
        print(f"Action: {'Rester' if action == 0 else 'Changer'}")
        print(f"Nouvel état: {state}")
        if done:
            print(f"Résultat: {'Gagné' if reward == 1 else 'Perdu'}")
            print(f"Récompense: {reward}")

if __name__ == "__main__":
    test_sarsa()