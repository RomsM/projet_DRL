import matplotlib
matplotlib.use('Agg')  # Utilise le backend Agg pour éviter les erreurs Qt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Assurez-vous d'avoir ces données après avoir exécuté l'algorithme Policy Iteration
# Ici, nous utilisons des exemples de données, vous devez les remplacer par vos propres résultats
policy = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0]
])
values = np.array([
    -6.80321585, -5.8617357,  -4.91074565, -3.95014964, -2.97985064,
    -5.8617357,   0.0,         -3.95014964, -2.97985064, -1.99975064,
    -4.91074565, -3.95014964,  0.0,         -1.99975064, -1.00975064,
    -3.95014964, -2.97985064, -1.99975064,  0.0,         -0.00975064,
    -2.97985064, -1.99975064, -1.00975064, -0.00975064, -0.00975064
])
average_rewards = np.random.normal(-50, 10, 100)  # Remplacez par vos propres résultats

# Create 'images' directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Helper function to plot a heatmap of the policy
def plot_policy(policy, title="Policy Iteration Policy Heatmap"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(policy, annot=True, cmap="viridis", cbar=False)
    plt.title(title)
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.savefig('visualisations/images/policy_iteration_policy_heatmap.png')
    plt.close()

# Helper function to plot a heatmap of the state values
def plot_state_values(values, title="Policy Iteration State Value Heatmap"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(values.reshape(5, 5), annot=True, cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.savefig('visualisations/images/policy_iteration_state_value_heatmap.png')
    plt.close()

# Helper function to plot average rewards per episode
def plot_average_rewards(rewards, title="Policy Iteration Average Rewards per Episode"):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Average Reward")
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig('visualisations/images/policy_iteration_average_rewards_per_episode.png')
    plt.close()

# Plotting the policy
plot_policy(policy)

# Plotting the state values
plot_state_values(values)

# Plotting the average rewards per episode
plot_average_rewards(average_rewards)
