import sys
import os

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.grid_world import GridWorld
from algorithms.on_policy_first_visit_mcc import OnPolicyFirstVisitMCC

def test_on_policy_first_visit_mcc():
    env = GridWorld(5, 5, (0, 0), (4, 4), [(1, 1), (2, 2), (3, 3)])
    agent = OnPolicyFirstVisitMCC(env)
    agent.train(num_episodes=100)
    policy = agent.get_policy()
    action_value_function = agent.get_action_value_function()

    print("Politique obtenue:")
    print(policy)
    print("Fonction de valeur-action obtenue:")
    print(action_value_function)

# Exécuter le test
if __name__ == "__main__":
    test_on_policy_first_visit_mcc()