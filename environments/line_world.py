class LineWorld:
    def __init__(self, length, start, goal):
        """
        Initialisation de l'environnement Line World.

        :param length: Longueur de la ligne (nombre d'états)
        :param start: État de départ (indice)
        :param goal: État objectif (indice)
        """
        self.length = length
        self.start = start
        self.goal = goal
        self.state = start
        self.actions = ['LEFT', 'RIGHT']

    def reset(self):
        """
        Réinitialise l'environnement à l'état de départ.

        :return: État initial
        """
        self.state = self.start
        return self.state

    def step(self, action):
        """
        Effectue une action et met à jour l'état de l'environnement.

        :param action: Action à effectuer ('LEFT' ou 'RIGHT')
        :return: Tuple contenant le nouvel état, la récompense et un booléen indiquant si l'état objectif est atteint
        """
        if action == 'LEFT':
            self.state = max(0, self.state - 1)
        elif action == 'RIGHT':
            self.state = min(self.length - 1, self.state + 1)
        
        reward = -1
        done = False
        if self.state == self.goal:
            reward = 0
            done = True

        return self.state, reward, done

    def render(self):
        """
        Affiche l'état actuel de l'environnement sous forme de ligne.
        """
        line = ['-'] * self.length
        line[self.state] = 'A'  # Agent
        line[self.goal] = 'G'   # Goal
        print(''.join(line))
