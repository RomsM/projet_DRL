import sys
import os
import numpy as np

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.grid_world import GridWorld
from environments.line_world import LineWorld
from environments.rock_paper_scissors import RockPaperScissors
from environments.monty_hall_level_1 import MontyHallLevel1
from environments.monty_hall_level_2 import MontyHallLevel2

from algorithms.policy_iteration import PolicyIteration
from algorithms.value_iteration import ValueIteration
from algorithms.monte_carlo_es import MonteCarloES
from algorithms.sarsa import SARSA
from algorithms.q_learning import QLearning
from algorithms.expected_sarsa import ExpectedSARSA
from algorithms.dyna_q import DynaQ

def test_algorithm_on_environment(algorithm_class, env_class, env_args, num_episodes=100):
    env = env_class(*env_args)
    agent = algorithm_class(env)
    
    # Vérifier les arguments attendus par la méthode train
    if algorithm_class in [PolicyIteration, ValueIteration]:
        agent.train()
    else:
        agent.train(num_episodes=num_episodes)
    
    policy = agent.get_policy()
    
    # Vérifier quelle méthode de valeur utiliser
    if hasattr(agent, 'get_action_value_function'):
        value_function = agent.get_action_value_function()
    elif hasattr(agent, 'get_value_function'):
        value_function = agent.get_value_function()
    else:
        value_function = None

    print(f"\nTest de {algorithm_class.__name__} sur {env_class.__name__}")
    print("Politique obtenue:")
    print(policy)
    print("Fonction de valeur-action obtenue:")
    print(value_function)

def main():
    environments = [
        (GridWorld, (5, 5, (0, 0), (4, 4), [(1, 1), (2, 2), (3, 3)])),
        (LineWorld, (10, 0, 9)),
        (RockPaperScissors, ()),
        (MontyHallLevel1, ()),
        (MontyHallLevel2, ())
    ]

    algorithms = [
        PolicyIteration,
        ValueIteration,
        MonteCarloES,
        SARSA,
        QLearning,
        ExpectedSARSA,
        DynaQ
    ]

    for env_class, env_args in environments:
        for algorithm_class in algorithms:
            test_algorithm_on_environment(algorithm_class, env_class, env_args, num_episodes=100)

if __name__ == "__main__":
    main()
