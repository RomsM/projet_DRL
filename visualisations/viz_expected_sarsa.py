import matplotlib
matplotlib.use('Agg')  # Utilise le backend Agg pour éviter les erreurs Qt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Assurez-vous d'avoir ces données après avoir exécuté l'algorithme Expected SARSA
# Ici, nous utilisons des exemples de données, vous devez les remplacer par vos propres résultats
policy = np.array([
    [0., 1., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [1., 0., 0., 0.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [0., 1., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
    [1., 0., 0., 0.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 1., 0.],
    [0., 1., 0., 0.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
    [0., 0., 0., 1.],
    [0., 0., 0., 1.],
    [1., 0., 0., 0.]
])
values = np.array([-4.76739938, -4.76278872, -4.77750871, -4.77343948, -4.21406186,
                   -4.11925339, -4.20001804, -4.12412196, -3.54657829, -3.47383111,
                   -3.46881996, -3.5022076, -2.95622056, -2.92092815, -3.01554204,
                   -2.94563465, -2.47982757, -2.51238582, -2.557468, -2.56429549,
                   -4.17626855, -4.15640091, -4.19900673, -4.21174101, 0.])
average_rewards = np.random.normal(-50, 10, 100)  # Remplacez par vos propres résultats

# Create 'images' directory if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# Helper function to plot a heatmap of the policy
def plot_policy(policy, title="Expected SARSA Policy Heatmap"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(policy, annot=True, cmap="viridis", cbar=False)
    plt.title(title)
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.savefig('images/expected_sarsa_policy_heatmap.png')
    plt.close()

# Helper function to plot a heatmap of the state values
def plot_state_values(values, title="Expected SARSA State Value Heatmap"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(values.reshape(5, 5), annot=True, cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.savefig('visualisations/images/expected_sarsa_state_value_heatmap.png')
    plt.close()

# Helper function to plot average rewards per episode
def plot_average_rewards(rewards, title="Expected SARSA Average Rewards per Episode"):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Average Reward")
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig('visualisations/images/expected_sarsa_average_rewards_per_episode.png')
    plt.close()

# Plotting the policy
plot_policy(policy)

# Plotting the state values
plot_state_values(values)

# Plotting the average rewards per episode
plot_average_rewards(average_rewards)
