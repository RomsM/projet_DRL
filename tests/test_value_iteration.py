import sys
import os

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.grid_world import GridWorld
from algorithms.value_iteration import ValueIteration

def test_value_iteration():
    env = GridWorld(5, 5, (0, 0), (4, 4), [(1, 1), (2, 2), (3, 3)])
    agent = ValueIteration(env)
    agent.train()
    policy = agent.get_policy()
    value_function = agent.get_value_function()

    print("Politique obtenue:")
    print(policy)
    print("Fonction de valeur obtenue:")
    print(value_function)

# Exécuter le test
if __name__ == "__main__":
    test_value_iteration()
