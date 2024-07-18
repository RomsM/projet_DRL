import sys
import os
import numpy as np

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.line_world import LineWorld
from algorithms.off_policy_mcc import OffPolicyMCC

def test_off_policy_mcc():
    # Créer l'environnement LineWorld
    env = LineWorld(length=10, start=0, goal=9)

    # Créer l'agent OffPolicyMCC
    agent = OffPolicyMCC(env)

    # Entraîner l'agent
    agent.train(num_episodes=1000)

    # Obtenir la politique et la fonction de valeur-action
    policy = agent.get_policy()
    action_value_function = agent.get_action_value_function()

    print("Politique obtenue:")
    print(policy)
    print("Fonction de valeur-action obtenue:")
    print(action_value_function)

    # Démonstration de la politique
    print("\nDémonstration de la politique:")
    state = env.reset()
    env.render()
    steps = 0
    while steps < 100:  # Limite pour éviter les boucles infinies
        action = int(policy[state])
        next_state, reward, done, _ = env.step(action)
        env.render()
        print(f"Action: {action}, État: {next_state}, Récompense: {reward}")
        state = next_state
        steps += 1
        if done:
            break
    if steps >= 100:
        print("Limite de pas atteinte, la politique peut ne pas être optimale.")

    # Interaction manuelle
    print("\nInteraction manuelle:")
    state = env.reset()
    env.render()
    while True:
        action = int(input("Entrez l'action (0 pour gauche, 1 pour droite): "))
        next_state, reward, done, _ = env.step(action)
        env.render()
        print(f"État: {next_state}, Récompense: {reward}")
        if done:
            break

# Exécuter le test
if __name__ == "__main__":
    test_off_policy_mcc()
