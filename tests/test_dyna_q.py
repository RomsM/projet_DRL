import sys
import os

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.grid_world import GridWorld
from algorithms.dyna_q import DynaQ

def test_dyna_q():
    env = GridWorld(5, 5, (0, 0), (4, 4), [(1, 1), (2, 2), (3, 3)])
    agent = DynaQ(env)
    agent.train(num_episodes=100)  # Réduire le nombre d'épisodes pour les tests initiaux
    policy = agent.get_policy()
    action_value_function = agent.get_action_value_function()

    print("Politique obtenue:")
    print(policy)
    print("Fonction de valeur-action obtenue:")
    print(action_value_function)

# Exécuter le test
if __name__ == "__main__":
    test_dyna_q()
