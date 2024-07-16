import matplotlib
matplotlib.use('Agg')  # Utilise le backend Agg pour éviter les erreurs Qt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assurez-vous d'avoir ces données après avoir exécuté l'algorithme Q-Learning
# Ici, nous utilisons des exemples de données, vous devez les remplacer par vos propres résultats
policy = np.array([
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [1., 0., 0., 0.],
    [1., 0., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [1., 0., 0., 0.],
    [1., 0., 0., 0.],
    [0., 0., 0., 1.],
    [0., 1., 0., 0.],
    [0., 1., 0., 0.],
    [0., 1., 0., 0.],
    [1., 0., 0., 0.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
    [0., 0., 0., 1.],
    [1., 0., 0., 0.]
])
values = np.array([-4.76220133, -4.70116462, -4.77549011, -4.76173608, -4.19260931,
                   -4.20484134, -4.13043464, -4.14383235, -3.50309664, -3.4905232,
                   -3.55654569, -3.45952518, -2.83775877, -2.86633021, -3.00531112,
                   -2.88743951, -2.46906775, -2.44623325, -2.50043106, -2.46525871,
                   -4.04037569, -4.07610795, -4.06801262, -4.09696349, 0.])
average_rewards = np.random.normal(-50, 10, 100)  # Remplacez par vos propres résultats

# Helper function to plot a heatmap of the policy
def plot_policy(policy, title="Q-Learning Policy Heatmap"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(policy, annot=True, cmap="viridis", cbar=False)
    plt.title(title)
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.savefig('q_learning_policy_heatmap.png')
    plt.close()

# Helper function to plot a heatmap of the state values
def plot_state_values(values, title="Q-Learning State Value Heatmap"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(values.reshape(5, 5), annot=True, cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.savefig('q_learning_state_value_heatmap.png')
    plt.close()

# Helper function to plot average rewards per episode
def plot_average_rewards(rewards, title="Q-Learning Average Rewards per Episode"):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Average Reward")
    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.savefig('q_learning_average_rewards_per_episode.png')
    plt.close()

# Plotting the policy
plot_policy(policy)

# Plotting the state values
plot_state_values(values)

# Plotting the average rewards per episode
plot_average_rewards(average_rewards)
