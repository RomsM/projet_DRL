import numpy as np

class GridWorld:
    def __init__(self, width, height, start, goal, obstacles=[]):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.state = start
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        x, y = self.state
        if action == 'UP':
            y = max(0, y - 1)
        elif action == 'DOWN':
            y = min(self.height - 1, y + 1)
        elif action == 'LEFT':
            x = max(0, x - 1)
        elif action == 'RIGHT':
            x = min(self.width - 1, x + 1)
        
        next_state = (x, y)
        if next_state in self.obstacles:
            next_state = self.state
        
        reward = -1
        if next_state == self.goal:
            reward = 0
        
        self.state = next_state
        return next_state, reward, next_state == self.goal
    
    def render(self):
        grid = np.zeros((self.height, self.width), dtype=int)
        for obs in self.obstacles:
            grid[obs[1], obs[0]] = -1
        grid[self.goal[1], self.goal[0]] = 1
        grid[self.state[1], self.state[0]] = 2
        print(grid)
