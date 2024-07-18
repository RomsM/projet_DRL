import sys
import os

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.line_world import LineWorld
from algorithms.expected_sarsa import ExpectedSarsa

def test_expected_sarsa():
    print("Initializing environment...")
    env = LineWorld(length=10, start=0, goal=9)
    print("Initializing agent...")
    agent = ExpectedSarsa(env)
    print("Starting training...")
    agent.train(num_episodes=1000)  # Réduire le nombre d'épisodes pour les tests
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
    max_steps = 100
    while state != env.goal and steps < max_steps:
        action = policy[state]
        state, reward, done, _ = env.step(action)
        env.render()
        print(f"Action: {action}, État: {state}, Récompense: {reward}")
        steps += 1
    if steps >= max_steps:
        print("Limite de pas atteinte, la politique peut ne pas être optimale.")

    # Interaction manuelle
    print("\nInteraction manuelle:")
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = int(input("Entrez l'action (0 pour gauche, 1 pour droite): "))
        state, reward, done, _ = env.step(action)
        env.render()
        print(f"État: {state}, Récompense: {reward}")

if __name__ == "__main__":
    test_expected_sarsa()
