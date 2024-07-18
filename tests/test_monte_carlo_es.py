import sys
import os
import numpy as np

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.line_world import LineWorld
from algorithms.monte_carlo_es import monte_carlo_es

def test_monte_carlo_es():
    # Définir l'environnement LineWorld
    length = 10  # Longueur de LineWorld
    start = 0  # Position de départ
    goal = 9  # Position d'objectif
    env = LineWorld(length, start, goal)
    
    # Exécuter l'algorithme Monte Carlo ES
    policy, value_function = monte_carlo_es(env, num_episodes=100, gamma=1.0)  # Réduire le nombre d'épisodes pour les tests

    print("Politique obtenue:")
    print(policy)
    print("Fonction de valeur obtenue:")
    print(value_function)

    # Démontrer la politique étape par étape
    print("\nDémonstration de la politique:")
    state = env.reset()
    done = False
    steps = 0
    max_steps = 100  # Limite maximale de pas pour éviter les boucles infinies
    while not done and steps < max_steps:
        env.render()
        action = int(policy[state])
        print(f"Action: {action}")
        state, reward, done, _ = env.step(action)
        print(f"État: {state}, Récompense: {reward}")
        steps += 1

    if steps >= max_steps:
        print("Limite de pas atteinte, la politique peut ne pas être optimale.")

    # Interaction manuelle
    print("\nInteraction manuelle:")
    env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        env.render()
        action = int(input("Entrez l'action (0 pour gauche, 1 pour droite): "))
        state, reward, done, _ = env.step(action)
        print(f"État: {state}, Récompense: {reward}")
        steps += 1

    if steps >= max_steps:
        print("Limite de pas atteinte lors de l'interaction manuelle.")

# Exécuter le test
if __name__ == "__main__":
    test_monte_carlo_es()
