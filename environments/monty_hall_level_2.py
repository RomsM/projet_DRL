import random

class MontyHallLevel2:
    def __init__(self):
        self.doors = ['A', 'B', 'C', 'D', 'E']
        self.winning_door = None
        self.chosen_door = None
        self.revealed_doors = []
        self.state = None
        self.step_count = 0

    def reset(self):
        """
        Réinitialise l'environnement à l'état initial.

        :return: État initial
        """
        self.winning_door = random.choice(self.doors)
        self.chosen_door = None
        self.revealed_doors = []
        self.state = 'CHOOSE_FIRST_DOOR'
        self.step_count = 0
        return self.state

    def step(self, action):
        """
        Effectue une action et met à jour l'état de l'environnement.

        :param action: Action à effectuer (choix de porte ou décision de changement)
        :return: Tuple contenant le nouvel état, la récompense et un booléen indiquant si l'état final est atteint
        """
        if self.state == 'CHOOSE_FIRST_DOOR':
            self.chosen_door = action
            self._reveal_door()
            self.state = 'DECIDE_CHANGE'
            self.step_count += 1
            return self.state, 0, False

        elif self.state == 'DECIDE_CHANGE':
            if action == 'CHANGE':
                self.chosen_door = [door for door in self.doors if door != self.chosen_door and door not in self.revealed_doors][0]
            self.step_count += 1
            if self.step_count < 4:
                self._reveal_door()
                return self.state, 0, False
            else:
                reward = 1 if self.chosen_door == self.winning_door else 0
                self.state = 'DONE'
                return self.state, reward, True

    def _reveal_door(self):
        """
        Révèle une porte non choisie et non gagnante.

        :return: Porte révélée
        """
        available_doors = [door for door in self.doors if door != self.chosen_door and door != self.winning_door and door not in self.revealed_doors]
        if available_doors:
            revealed_door = random.choice(available_doors)
            self.revealed_doors.append(revealed_door)

    def render(self):
        """
        Affiche l'état actuel de l'environnement.
        """
        if self.state == 'CHOOSE_FIRST_DOOR':
            print("Choisissez une porte: A, B, C, D ou E")
        elif self.state == 'DECIDE_CHANGE':
            print(f"Vous avez choisi la porte {self.chosen_door}. Les portes révélées sont : {', '.join(self.revealed_doors)}.")
            print("Voulez-vous changer votre choix ? (ACTION: 'CHANGE' ou 'KEEP')")
        elif self.state == 'DONE':
            result = "gagné" if self.chosen_door == self.winning_door else "perdu"
            print(f"La porte gagnante était {self.winning_door}. Vous avez {result}!")
