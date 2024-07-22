import sys
import os

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from environments.monty_hall_level_1 import MontyHallLevel1
from algorithms.policy_iteration import PolicyIteration

def test_policy_iteration():
    print("Initializing environment...")
    env = MontyHallLevel1()
    print("Initializing agent...")
    agent = PolicyIteration(env)
    print("Starting training...")
    agent.train()  # Policy Iteration n'a pas besoin de spécifier le nombre d'épisodes
    policy = agent.get_policy()
    value_function = agent.get_value_function()  # Policy Iteration utilise une fonction de valeur, pas une fonction de valeur-action

    print("Politique obtenue:")
    print(policy)
    print("Fonction de valeur obtenue:")
    print(value_function)

    # Sauvegarde de la politique et des fonctions
    agent.save(r'D:\projet_DRL\tests\Monty_hall_Lvl_1\policy\policy_iteration_MH1.npz')
    print("Politique et fonctions sauvegardées dans 'policy_iteration_MH1.npz'.")

    # Chargement de la politique et des fonctions
    agent.load(r'D:\projet_DRL\tests\Monty_hall_Lvl_1\policy\policy_iteration_MH1.npz')
    loaded_policy = agent.get_policy()
    loaded_value_function = agent.get_value_function()
    print("Politique chargée:")
    print(loaded_policy)
    print("Fonction de valeur chargée:")
    print(loaded_value_function)

    # Démonstration de la politique
    print("\nDémonstration de la politique:")
    num_episodes = 1000
    total_wins = 0
    for _ in range(num_episodes):
        state = env.reset()
        action = policy[state]
        _, reward, done, _ = env.step(action)
        total_wins += reward
    win_rate = total_wins / num_episodes
    print(f"Taux de victoire sur {num_episodes} épisodes: {win_rate:.2%}")

    # Interaction manuelle
    print("\nInteraction manuelle:")
    state = env.reset()
    print(f"État initial: {state}")
    done = False
    while not done:
        action = int(input("Entrez l'action (0 pour rester, 1 pour changer): "))
        state, reward, done, _ = env.step(action)
        print(f"Action: {'Rester' if action == 0 else 'Changer'}")
        print(f"Résultat: {'Gagné' if reward == 1 else 'Perdu'}")
        print(f"Récompense: {reward}")

if __name__ == "__main__":
    test_policy_iteration()