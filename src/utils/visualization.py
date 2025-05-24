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

def plot_dag_with_locations(dag, service_locations, num_nodes, filename=None):
    """Visualize the DAG structure and service locations on nodes."""
    pos = nx.spring_layout(dag)
    node_colors = [service_locations[n] for n in dag.nodes]
    cmap = plt.get_cmap('tab10', num_nodes)
    nx.draw(dag, pos, with_labels=True, node_color=node_colors, cmap=cmap, node_size=500)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_nodes-1))
    plt.colorbar(sm, ticks=range(num_nodes), label='Edge Node ID')
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def save_migration_animation(dag, service_locs_history, num_nodes, gif_filename='migration_animation.gif'):
    """Create a GIF animation of service migration over time."""
    images = []
    for step, service_locations in enumerate(service_locs_history):
        fname = f'dag_step_{step}.png'
        plot_dag_with_locations(dag, service_locations, num_nodes, filename=fname)
        images.append(imageio.imread(fname))
    imageio.mimsave(gif_filename, images, duration=0.5)

def export_rewards_to_csv(rewards, filename='rewards.csv'):
    """Export rewards to a CSV file."""
    pd.DataFrame({'reward': rewards}).to_csv(filename, index=False)

def export_service_locations_history(service_locs_history, filename='service_locations_history.csv'):
    """Export service locations history to a CSV file."""
    pd.DataFrame(service_locs_history).to_csv(filename, index=False)

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