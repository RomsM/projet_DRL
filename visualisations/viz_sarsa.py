import matplotlib
matplotlib.use('Agg')  # Utilise le backend Agg pour éviter les erreurs Qt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Assurez-vous d'avoir ces données après avoir exécuté l'algorithme SARSA
# Ici, nous utilisons des exemples de données, vous devez les remplacer par vos propres résultats
policy = np.array([
    [0., 1., 0., 0.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [1., 0., 0., 0.],
    [1., 0., 0., 0.],
    [0., 0., 0., 1.],
    [1., 0., 0., 0.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [1., 0., 0., 0.],
    [0., 0., 0., 1.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [1., 0., 0., 0.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
    [0., 0., 0., 1.],
    [1., 0., 0., 0.]
])
values = np.array([-4.80887627, -4.77225084, -4.88270056, -4.80037792, -4.12266605,
                   -4.27513215, -4.15918848, -4.14333152, -3.50853369, -3.45522524,
                   -3.55615244, -3.47968515, -2.91543632, -2.89072561, -2.95819655,
                   -2.86815656, -2.49585784, -2.44627998, -2.40838626, -2.57717483,
                   -4.2455064 , -4.1212637 , -4.14788657, -4.17182119, 0.])
average_rewards = np.random.normal(-50, 10, 100)  # Remplacez par vos propres résultats

# Create 'images' directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Helper function to plot a heatmap of the policy
def plot_policy(policy, title="SARSA Policy Heatmap"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(policy, annot=True, cmap="viridis", cbar=False)
    plt.title(title)
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.savefig('images/sarsa_policy_heatmap.png')
    plt.close()

# Helper function to plot a heatmap of the state values
def plot_state_values(values, title="SARSA State Value Heatmap"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(values.reshape(5, 5), annot=True, cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.savefig('images/sarsa_state_value_heatmap.png')
    plt.close()

# Helper function to plot average rewards per episode
def plot_average_rewards(rewards, title="SARSA Average Rewards per Episode"):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Average Reward")
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig('images/sarsa_average_rewards_per_episode.png')
    plt.close()

# Plotting the policy
plot_policy(policy)

# Plotting the state values
plot_state_values(values)

# Plotting the average rewards per episode
plot_average_rewards(average_rewards)
