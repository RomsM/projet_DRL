import sys
import os

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.monty_hall_level_1 import MontyHallLevel1

def test_monty_hall_level_1():
    env = MontyHallLevel1()

    # Réinitialiser l'environnement et afficher l'état initial
    state = env.reset()
    print("État initial:")
    env.render()

    # Choisir une porte et décider de changer ou non
    actions = ['A', 'CHANGE']
    for action in actions:
        next_state, reward, done = env.step(action)
        print(f"Action: {action}")
        print(f"Nouvel état: {next_state}, Récompense: {reward}, Terminé: {done}")
        env.render()
        if done:
            print("La partie est terminée!")
            break

# Exécuter le test
if __name__ == "__main__":
    test_monty_hall_level_1()
