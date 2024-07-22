import matplotlib

matplotlib.use('Agg')  # Utiliser le backend Agg pour éviter les problèmes liés à Qt
import numpy as np
import matplotlib.pyplot as plt


def analyze_results():
    # Chargement des résultats
    data = np.load(r'D:\projet_DRL\tests\Monty_Hall_Lvl_1\policy\experiment_results.npz', allow_pickle=True)

    algorithms = [
        'Dyna-Q',
        'Q-Learning',
        'SARSA',
        'Monte Carlo ES',
        'Off-Policy MCC',
        'On-Policy First Visit MCC',
        'Policy Iteration',
        'Value Iteration'
    ]

    rewards = []
    durations = []

    for algo in algorithms:
        rewards.append(data[algo].item().get('total_reward'))
        durations.append(data[algo].item().get('duration'))

    # Affichage des résultats
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.barh(algorithms, rewards, color='skyblue')
    plt.xlabel('Total Reward')
    plt.title('Total Rewards per Algorithm')

    plt.subplot(1, 2, 2)
    plt.barh(algorithms, durations, color='lightgreen')
    plt.xlabel('Duration (seconds)')
    plt.title('Training Duration per Algorithm')

    plt.tight_layout()
    plt.savefig('algorithm_performance_comparison.png')
    plt.show()


if __name__ == "__main__":
    analyze_results()
