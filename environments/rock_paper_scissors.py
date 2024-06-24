import random

class RockPaperScissors:
    def __init__(self):
        self.actions = ['ROCK', 'PAPER', 'SCISSORS']
        self.state = None
        self.first_round_action = None

    def reset(self):
        """
        Réinitialise l'environnement à l'état initial.

        :return: État initial
        """
        self.state = 'FIRST_ROUND'
        self.first_round_action = None
        return self.state

    def step(self, action):
        """
        Effectue une action et met à jour l'état de l'environnement.

        :param action: Action à effectuer ('ROCK', 'PAPER' ou 'SCISSORS')
        :return: Tuple contenant le nouvel état, la récompense et un booléen indiquant si l'état final est atteint
        """
        if self.state == 'FIRST_ROUND':
            opponent_action = random.choice(self.actions)
            reward = self._get_reward(action, opponent_action)
            self.first_round_action = action
            self.state = 'SECOND_ROUND'
            return self.state, reward, False

        elif self.state == 'SECOND_ROUND':
            opponent_action = self.first_round_action
            reward = self._get_reward(action, opponent_action)
            self.state = 'DONE'
            return self.state, reward, True

    def _get_reward(self, action, opponent_action):
        """
        Calcule la récompense en fonction des actions de l'agent et de l'adversaire.

        :param action: Action de l'agent
        :param opponent_action: Action de l'adversaire
        :return: Récompense (1 pour victoire, -1 pour défaite, 0 pour égalité)
        """
        if action == opponent_action:
            return 0
        elif (action == 'ROCK' and opponent_action == 'SCISSORS') or \
             (action == 'PAPER' and opponent_action == 'ROCK') or \
             (action == 'SCISSORS' and opponent_action == 'PAPER'):
            return 1
        else:
            return -1

    def render(self):
        """
        Affiche l'état actuel de l'environnement.
        """
        if self.state == 'FIRST_ROUND':
            print("Premier round: choisissez ROCK, PAPER ou SCISSORS")
        elif self.state == 'SECOND_ROUND':
            print(f"Premier round terminé. Votre action: {self.first_round_action}.")
            print("Deuxième round: choisissez ROCK, PAPER ou SCISSORS")
        elif self.state == 'DONE':
            print("Partie terminée.")
