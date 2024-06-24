import random

class MontyHallLevel1:
    def __init__(self):
        self.doors = ['A', 'B', 'C']
        self.winning_door = None
        self.chosen_door = None
        self.revealed_door = None
        self.state = None

    def reset(self):
        """
        Réinitialise l'environnement à l'état initial.

        :return: État initial
        """
        self.winning_door = random.choice(self.doors)
        self.chosen_door = None
        self.revealed_door = None
        self.state = 'CHOOSE_FIRST_DOOR'
        return self.state

    def step(self, action):
        """
        Effectue une action et met à jour l'état de l'environnement.

        :param action: Action à effectuer (choix de porte ou décision de changement)
        :return: Tuple contenant le nouvel état, la récompense et un booléen indiquant si l'état final est atteint
        """
        if self.state == 'CHOOSE_FIRST_DOOR':
            self.chosen_door = action
            self.revealed_door = self._reveal_door()
            self.state = 'DECIDE_CHANGE'
            return self.state, 0, False

        elif self.state == 'DECIDE_CHANGE':
            if action == 'CHANGE':
                self.chosen_door = [door for door in self.doors if door != self.chosen_door and door != self.revealed_door][0]
            reward = 1 if self.chosen_door == self.winning_door else 0
            self.state = 'DONE'
            return self.state, reward, True

    def _reveal_door(self):
        """
        Révèle une porte non choisie et non gagnante.

        :return: Porte révélée
        """
        available_doors = [door for door in self.doors if door != self.chosen_door and door != self.winning_door]
        return random.choice(available_doors)

    def render(self):
        """
        Affiche l'état actuel de l'environnement.
        """
        if self.state == 'CHOOSE_FIRST_DOOR':
            print("Choisissez une porte: A, B ou C")
        elif self.state == 'DECIDE_CHANGE':
            print(f"Vous avez choisi la porte {self.chosen_door}. La porte {self.revealed_door} est révélée et n'est pas gagnante.")
            print("Voulez-vous changer votre choix ? (ACTION: 'CHANGE' ou 'KEEP')")
        elif self.state == 'DONE':
            result = "gagné" if self.chosen_door == self.winning_door else "perdu"
            print(f"La porte gagnante était {self.winning_door}. Vous avez {result}!")
