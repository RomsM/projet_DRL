from gym.spaces import Discrete

class LineWorld:
    def __init__(self, length, start, goal):
        self.length = length
        self.start = start
        self.goal = goal
        self.current_state = start
        self.observation_space = Discrete(length)  # Ajouter l'attribut observation_space
        self.action_space = Discrete(2)  # Ajouter l'attribut action_space pour 2 actions possibles (gauche, droite)
        self.P = self._build_transition_probabilities()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        if action == 0:  # gauche
            self.state = max(0, self.state - 1)
        elif action == 1:  # droite
            self.state = min(self.length - 1, self.state + 1)

        reward = -0.01
        done = False
        if self.state == self.goal:
            reward = 1.0
            done = True

        return self.state, reward, done, {}

    def render(self):
        world = ['-'] * self.length
        world[self.goal] = 'G'
        world[self.state] = 'O'
        print(''.join(world))

    def _build_transition_probabilities(self):
        P = {state: {action: [] for action in range(self.action_space.n)} for state in range(self.observation_space.n)}
        for state in range(self.observation_space.n):
            for action in range(self.action_space.n):
                if action == 0:  # gauche
                    next_state = max(0, state - 1)
                else:  # droite
                    next_state = min(self.length - 1, state + 1)
                reward = 1.0 if next_state == self.goal else -0.01
                done = next_state == self.goal
                P[state][action].append((1.0, next_state, reward, done))
        return P

    @property
    def observation_space_size(self):
        return self.length

    @property
    def action_space_size(self):
        return 2