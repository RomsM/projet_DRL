import sys
import os

# Ajouter le chemin racine de votre projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.rock_paper_scissors import RockPaperScissors

def test_rock_paper_scissors():
    env = RockPaperScissors()

    # Réinitialiser l'environnement et afficher l'état initial
    state = env.reset()
    print("État initial:")
    env.render()

    # Effectuer quelques actions et afficher les résultats
    actions = ['ROCK', 'PAPER']
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
    test_rock_paper_scissors()

