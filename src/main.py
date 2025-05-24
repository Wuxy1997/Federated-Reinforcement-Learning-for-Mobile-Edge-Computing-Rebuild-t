import torch
import numpy as np
import copy
import random
import gym # Import gym
import matplotlib.pyplot as plt # Import matplotlib.pyplot
from environment.mec_env import MECEnvironment
from agents.dqn_agent import DQNAgent
from federated.federated_learning import FederatedLearning
from robust.robust_federated_learning import RobustFederatedLearning
from utils.visualization import plot_multiple_rewards, plot_reward_histogram, plot_dag_with_locations, save_migration_animation, export_rewards_to_csv, export_service_locations_history, plot_multi_metrics
from environment.microservice_migration_env import MicroserviceMigrationEnv # Import the migration environment

# User configurable parameters
USE_ROBUST_FL = True  # True: Use robust federated learning, False: Standard federated learning
AGGREGATION_METHOD = 'krum'  # 'krum', 'median', 'trimmed_mean' (only effective for robust FL)
NUM_ADV = 1  # Number of adversarial clients (only for krum)
TRIM_RATIO = 0.1  # Trim ratio for trimmed mean (only for trimmed_mean)

# ==== Model poisoning attack parameters ====
ENABLE_ATTACK = True  # Whether to enable model poisoning attack
ATTACK_METHOD = 'random_noise'  # 'random_noise', 'sign_flip', etc.
ATTACK_EPSILON = 10.0  # Attack strength (e.g., noise magnitude)

# ==== Environment Parameters ====
USE_MIGRATION_ENV = False # True: Use MicroserviceMigrationEnv, False: Use MECEnvironment
NUM_EDGE_NODES = 3
NUM_MOBILE_DEVICES = 5 # Also number of clients in FL
NUM_SERVICES = 5 # Number of microservices for migration env

# ==== Training Parameters ====
EPISODES_PER_ROUND = 100
NUM_ROUNDS = 10


def poison_model(state_dict, method='random_noise', epsilon=10.0):
    poisoned = copy.deepcopy(state_dict)
    for k in poisoned.keys():
        if method == 'random_noise':
            poisoned[k] += epsilon * torch.randn_like(poisoned[k])
        elif method == 'sign_flip':
            poisoned[k] = -poisoned[k]
        # You can extend more attack methods here
    return poisoned

def train_client(agent, env, episodes=100):
    """Train a single client agent"""
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            # Ensure action is in correct format for MultiDiscrete if using migration env
            if isinstance(env.action_space, gym.spaces.MultiDiscrete):
                 # Agent action is an index, need to map to (service_id, target_node_id)
                 # This requires a proper mapping based on your agent's output layer
                 # For now, let's assume the agent directly outputs the multi-discrete action tuple
                 pass # Agent should output a tuple (service_id, target_node)
            
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
        
        rewards_history.append(total_reward)
        
        if episode % 10 == 0:
            agent.update_target_network()
    
    return rewards_history

def main():

    # Create environment
    if USE_MIGRATION_ENV:
        env = MicroserviceMigrationEnv(num_nodes=NUM_EDGE_NODES, num_services=NUM_SERVICES)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.nvec.prod() # For simple flattened multi-discrete
        # NOTE: For complex MultiDiscrete action space, agent and action handling need careful design.
        # The current simple DQN and action_size calculation might not be suitable.
        # This is a placeholder.
        initial_dag = env.dag # Store initial DAG for visualization
    else:
        env = MECEnvironment(num_edge_nodes=NUM_EDGE_NODES, num_mobile_devices=NUM_MOBILE_DEVICES)
        state_size = 5  # [device_computation, edge_computation, task_size, task_deadline, channel_state]
        action_size = NUM_EDGE_NODES + 1  # +1 for local execution
    
    # Select federated learning framework
    if USE_ROBUST_FL:
        federated_learning = RobustFederatedLearning(num_clients=NUM_MOBILE_DEVICES, aggregation_rounds=NUM_ROUNDS)
        print(f"Using Robust Federated Learning with aggregation method: {AGGREGATION_METHOD}")
    else:
        federated_learning = FederatedLearning(num_clients=NUM_MOBILE_DEVICES, aggregation_rounds=NUM_ROUNDS)
        print("Using Standard Federated Learning (FedAvg)")
    
    # Initialize agents for each mobile device
    # NOTE: If using MicroserviceMigrationEnv with complex action space, DQNAgent needs modification.
    agents = []
    for _ in range(NUM_MOBILE_DEVICES):
        agent = DQNAgent(state_size=state_size, action_size=action_size)
        agents.append(agent)
    
    # Initialize federated learning with the first agent's model
    federated_learning.initialize_global_model(agents[0].q_network)
    federated_learning.initialize_client_models(agents[0].q_network)
    
    # Training history
    global_rewards = []
    # Add other metrics to track if needed, e.g., total migrations per round
    # total_migrations_per_round = []
    
    # Federated learning rounds
    for round in range(NUM_ROUNDS):
        print(f"\nFederated Learning Round {round + 1}/{NUM_ROUNDS}")
        
        # Train each client
        client_rewards = []
        # client_migration_counts = [] # Example for multi-metrics

        for i, agent in enumerate(agents):
            print(f"Training client {i + 1}/{NUM_MOBILE_DEVICES}")
            rewards = train_client(agent, env, episodes=EPISODES_PER_ROUND)
            client_rewards.append(rewards)
            # migration_count = calculate_migrations(env, agent) # Example
            # client_migration_counts.append(migration_count)
        
        # Update global model
        client_weights = [agent.q_network.state_dict() for agent in agents]
        client_sizes = [len(rewards) for rewards in client_rewards]

        # ==== Model poisoning attack simulation ====
        if ENABLE_ATTACK and NUM_ADV > 0:
            adv_indices = random.sample(range(len(client_weights)), NUM_ADV)
            for idx in adv_indices:
                client_weights[idx] = poison_model(client_weights[idx], method=ATTACK_METHOD, epsilon=ATTACK_EPSILON)
            print(f"Injected poisoning to clients: {adv_indices}")

        # ==== Aggregation ====
        if USE_ROBUST_FL:
            kwargs = {}
            if AGGREGATION_METHOD == 'krum':
                kwargs['num_adv'] = NUM_ADV
            elif AGGREGATION_METHOD == 'trimmed_mean':
                kwargs['trim_ratio'] = TRIM_RATIO
            federated_learning.update_global_model(client_weights, client_sizes, method=AGGREGATION_METHOD, **kwargs)
        else:
            federated_learning.update_global_model(client_weights, client_sizes)
        
        # Distribute global model to clients
        federated_learning.distribute_global_model()
        for i, agent in enumerate(agents):
            agent.q_network.load_state_dict(federated_learning.global_model.state_dict())
            agent.target_network.load_state_dict(agent.q_network.state_dict())
        
        # Record average reward for this round
        avg_reward = np.mean([np.mean(rewards) for rewards in client_rewards])
        global_rewards.append(avg_reward)
        print(f"Round {round + 1} Average Reward: {avg_reward:.2f}")

        # total_migrations_per_round.append(np.sum(client_migration_counts)) # Example
    
    # ==== Visualization and Data Export ====

    # 1. Plot Global Average Reward Curve
    plt.figure(figsize=(10, 6))
    plt.plot(global_rewards, label='Global Average Reward')
    plt.xlabel('Federated Learning Round')
    plt.ylabel('Average Reward')
    plt.title('Federated Reinforcement Learning Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_results.png')
    plt.close()
    
    # 2. Plot Reward Distribution Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(global_rewards, bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('Average Reward')
    plt.ylabel('Frequency')
    plt.title('Distribution of Global Average Rewards')
    plt.grid(True, alpha=0.3)
    plt.savefig('reward_distribution.png')
    plt.close()

    # 3. Plot Client-wise Reward Comparison
    plt.figure(figsize=(12, 6))
    client_avg_rewards = [np.mean(rewards) for rewards in client_rewards]
    plt.bar(range(len(client_avg_rewards)), client_avg_rewards, color='lightgreen')
    plt.xlabel('Client ID')
    plt.ylabel('Average Reward')
    plt.title('Average Rewards per Client')
    plt.xticks(range(len(client_avg_rewards)))
    plt.grid(True, alpha=0.3)
    plt.savefig('client_rewards.png')
    plt.close()

    # 4. Plot Service Location Heatmap (if using migration environment)
    if USE_MIGRATION_ENV:
        # Create a heatmap of service locations
        service_locations = np.zeros((NUM_SERVICES, NUM_EDGE_NODES))
        for i, agent in enumerate(agents):
            state = env.reset()
            service_locations += state[:NUM_SERVICES].reshape(NUM_SERVICES, NUM_EDGE_NODES)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(service_locations, cmap='YlOrRd')
        plt.colorbar(label='Service Count')
        plt.xlabel('Edge Node')
        plt.ylabel('Service ID')
        plt.title('Service Distribution Across Edge Nodes')
        plt.savefig('service_distribution.png')
        plt.close()

    # 5. Plot Training Loss (if available)
    if hasattr(agents[0], 'loss_history'):
        plt.figure(figsize=(10, 6))
        for i, agent in enumerate(agents):
            plt.plot(agent.loss_history, label=f'Client {i+1}', alpha=0.7)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss per Client')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.close()

    # 6. Plot Attack Impact Analysis (if attack is enabled)
    if ENABLE_ATTACK:
        # Calculate reward differences between rounds
        reward_diffs = np.diff(global_rewards)
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(global_rewards)), reward_diffs, 'r-', label='Reward Change')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.xlabel('Federated Learning Round')
        plt.ylabel('Reward Change')
        plt.title('Impact of Model Poisoning Attacks on Training')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('attack_impact.png')
        plt.close()

        # Plot attack success rate
        if hasattr(federated_learning, 'attack_success_history'):
            plt.figure(figsize=(10, 6))
            plt.plot(federated_learning.attack_success_history, 'b-', label='Attack Success Rate')
            plt.xlabel('Federated Learning Round')
            plt.ylabel('Success Rate')
            plt.title('Model Poisoning Attack Success Rate')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig('attack_success_rate.png')
            plt.close()

    # 7. Plot Aggregation Method Comparison
    if USE_ROBUST_FL:
        plt.figure(figsize=(10, 6))
        plt.plot(global_rewards, label=f'{AGGREGATION_METHOD} Aggregation', color='blue')
        # Add baseline (FedAvg) if available
        if hasattr(federated_learning, 'baseline_rewards'):
            plt.plot(federated_learning.baseline_rewards, label='FedAvg Baseline', color='red', linestyle='--')
        plt.xlabel('Federated Learning Round')
        plt.ylabel('Average Reward')
        plt.title('Robust vs Standard Aggregation Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig('aggregation_comparison.png')
        plt.close()

    # 8. Plot Client Performance Evolution
    plt.figure(figsize=(12, 6))
    for i, rewards in enumerate(client_rewards):
        plt.plot(rewards, label=f'Client {i+1}', alpha=0.7)
    plt.xlabel('Training Episode')
    plt.ylabel('Reward')
    plt.title('Client Performance Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('client_evolution.png')
    plt.close()

    # Export Rewards to CSV
    export_rewards_to_csv(global_rewards, filename='training_rewards.csv')
    print("Training rewards exported to training_rewards.csv")

    # Plot Initial and Final DAG Service Locations (only if using migration env)
    if USE_MIGRATION_ENV:
         print("Visualizing initial DAG structure...")
         env_for_plotting = MicroserviceMigrationEnv(num_nodes=NUM_EDGE_NODES, num_services=NUM_SERVICES)
         plot_dag_with_locations(initial_dag, env_for_plotting.reset()[:env_for_plotting.num_services], env_for_plotting.num_nodes, filename='initial_dag_locations.png')
         print("Initial DAG visualization saved to initial_dag_locations.png")

    # Save the final global model
    torch.save(federated_learning.global_model.state_dict(), 'global_model.pth')
    print("Final global model saved to global_model.pth")

    print("\nVisualization files generated:")
    print("1. training_results.png - Global average reward curve")
    print("2. reward_distribution.png - Reward distribution histogram")
    print("3. client_rewards.png - Client-wise reward comparison")
    if USE_MIGRATION_ENV:
        print("4. service_distribution.png - Service location heatmap")
    if hasattr(agents[0], 'loss_history'):
        print("5. training_loss.png - Training loss curves")
    if ENABLE_ATTACK:
        print("6. attack_impact.png - Impact of model poisoning attacks")
        if hasattr(federated_learning, 'attack_success_history'):
            print("7. attack_success_rate.png - Attack success rate over time")
    if USE_ROBUST_FL:
        print("8. aggregation_comparison.png - Robust vs standard aggregation comparison")
    print("9. client_evolution.png - Client performance evolution")
    print("10. training_rewards.csv - Detailed reward data")
    if USE_MIGRATION_ENV:
        print("11. initial_dag_locations.png - Initial DAG structure")

if __name__ == "__main__":
    main() 