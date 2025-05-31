import matplotlib.pyplot as plt
import networkx as nx
import imageio
import pandas as pd
import numpy as np

def plot_multiple_rewards(rewards_dict, filename='reward_comparison.png'):
    """Plot multiple reward curves for comparison."""
    plt.figure(figsize=(10, 6))
    for label, rewards in rewards_dict.items():
        plt.plot(rewards, label=label)
    plt.xlabel('Federated Learning Round')
    plt.ylabel('Average Reward')
    plt.title('Reward Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_reward_histogram(rewards, filename='reward_histogram.png'):
    """Plot a histogram of rewards."""
    plt.figure()
    plt.hist(rewards, bins=10)
    plt.xlabel('Average Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Average Rewards')
    plt.savefig(filename)
    plt.close()

def plot_dag_with_locations(dag, service_locations, num_nodes, filename='dag_locations.png'):
    """Visualize DAG with service locations (node colors)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(dag, seed=42)
    cmap = plt.get_cmap('tab10', num_nodes)
    node_colors = [cmap(int(service_locations[i])) for i in range(len(service_locations))]
    nx.draw_networkx_edges(dag, pos, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=20)
    nx.draw_networkx_nodes(dag, pos, node_color=node_colors, node_size=500, ax=ax)
    nx.draw_networkx_labels(dag, pos, ax=ax)
    norm = plt.Normalize(vmin=0, vmax=num_nodes-1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, ticks=range(num_nodes), label='Edge Node ID')
    plt.title('DAG Service Locations')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_migration_animation(frames, filename='migration_animation.gif', duration=0.5):
    imageio.mimsave(filename, frames, duration=duration)

def export_rewards_to_csv(rewards, filename='rewards.csv'):
    """Export rewards to a CSV file."""
    pd.DataFrame({'reward': rewards}).to_csv(filename, index=False)

def export_service_locations_history(history, filename='service_locations_history.csv'):
    pd.DataFrame(history).to_csv(filename, index=False)

def plot_multi_metrics(metrics_dict, filename='multi_metrics.png'):
    """Plot multiple metrics (e.g., reward, migration count) in subplots."""
    num_metrics = len(metrics_dict)
    plt.figure(figsize=(6 * num_metrics, 5))
    for i, (label, values) in enumerate(metrics_dict.items()):
        plt.subplot(1, num_metrics, i + 1)
        plt.plot(values, label=label)
        plt.xlabel('Round')
        plt.ylabel(label)
        plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() 