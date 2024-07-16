import matplotlib
matplotlib.use('Agg')  # Utilise le backend Agg pour éviter les erreurs Qt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Assurez-vous d'avoir ces données après avoir exécuté l'algorithme Monte Carlo ES
# Ici, nous utilisons des exemples de données, vous devez les remplacer par vos propres résultats
policy = np.array([
    [0., 0., 0., 1.],
    [0., 0., 0., 1.],
    [0., 1., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [1., 0., 0., 0.],
    [0.25, 0.25, 0.25, 0.25],
    [0., 0., 0., 1.],
    [0., 0., 0., 1.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [1., 0., 0., 0.],
    [0.25, 0.25, 0.25, 0.25],
    [0., 0., 0., 1.],
    [1., 0., 0., 0.],
    [1., 0., 0., 0.],
    [0., 0., 0., 1.],
    [0., 0., 0., 1.],
    [0.25, 0.25, 0.25, 0.25],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
    [1., 0., 0., 0.],
    [0., 0., 1., 0.],
    [1., 0., 0., 0.],
    [0.25, 0.25, 0.25, 0.25]
])
values = np.array([-59.52680273, -46.37317748, -60.33221936, -18.20930624, -37.64746051,
                   -48.48628826, -61.51039211, -60.72889716, -47.44035125, -43.03987975,
                   -47.96594773, 0., -53.41192248, -62.27633531, -63.02703624,
                   -62.65357195, -55.69520184, -61.89528819, -54.33902523, -57.44098766,
                   -76.00931751, -49.17524348, -74.93952032, -74.58497348, -96.4142412])
average_rewards = np.random.normal(-50, 10, 100)  # Remplacez par vos propres résultats

# Helper function to plot a heatmap of the policy
def plot_policy(policy, title="Policy Heatmap"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(policy, annot=True, cmap="viridis", cbar=False)
    plt.title(title)
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.savefig('policy_heatmap.png')
    plt.close()

# Helper function to plot a heatmap of the state values
def plot_state_values(values, title="State Value Heatmap"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(values.reshape(5, 5), annot=True, cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.savefig('state_value_heatmap.png')
    plt.close()

# Helper function to plot average rewards per episode
def plot_average_rewards(rewards, title="Average Rewards per Episode"):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Average Reward")
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig('average_rewards_per_episode.png')
    plt.close()

# Plotting the policy
plot_policy(policy)

# Plotting the state values
plot_state_values(values)

# Plotting the average rewards per episode
plot_average_rewards(average_rewards)
