import sys
import os

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.line_world import LineWorld

def test_line_world():
    length = 10
    start = 0
    goal = 9

    env = LineWorld(length, start, goal)

    # Réinitialiser l'environnement et afficher l'état initial
    state = env.reset()
    print("État initial:")
    env.render()

    # Effectuer quelques actions et afficher les résultats
    actions = ['RIGHT', 'RIGHT', 'RIGHT', 'LEFT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT', 'RIGHT']
    for action in actions:
        next_state, reward, done = env.step(action)
        print(f"Action: {action}")
        print(f"Nouvel état: {next_state}, Récompense: {reward}, Terminé: {done}")
        env.render()
        if done:
            print("L'agent a atteint l'état objectif!")
            break

# Exécuter le test
if __name__ == "__main__":
    test_line_world()
