from environments.grid_world import GridWorld

def test_gridworld():
    width = 5
    height = 5
    start = (0, 0)
    goal = (4, 4)
    obstacles = [(1, 1), (2, 2), (3, 3)]
    
    env = GridWorld(width, height, start, goal, obstacles)
    
    state = env.reset()
    print("État initial:")
    env.render()
    
    actions = ['RIGHT', 'RIGHT', 'DOWN', 'DOWN', 'LEFT', 'UP', 'DOWN', 'DOWN', 'RIGHT', 'RIGHT', 'RIGHT']
    for action in actions:
        next_state, reward, done = env.step(action)
        print(f"Action: {action}")
        print(f"Nouvel état: {next_state}, Récompense: {reward}, Terminé: {done}")
        env.render()
        if done:
            break

# Exécuter le test
if __name__ == "__main__":
    test_gridworld()
