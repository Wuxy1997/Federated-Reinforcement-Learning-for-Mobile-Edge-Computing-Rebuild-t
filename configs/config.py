# Environment Configuration
ENV_CONFIG = {
    'num_edge_nodes': 3,
    'num_mobile_devices': 5,
    'state_size': 5,  # [device_computation, edge_computation, task_size, task_deadline, channel_state]
    'action_size': 4,  # num_edge_nodes + 1 (local execution)
}

# Agent Configuration
AGENT_CONFIG = {
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995,
    'memory_size': 10000,
    'batch_size': 64
}

# Federated Learning Configuration
FEDERATED_CONFIG = {
    'num_clients': 5,
    'aggregation_rounds': 10,
    'episodes_per_round': 100,
    'num_rounds': 10
}

# Training Configuration
TRAINING_CONFIG = {
    'save_interval': 10,  # Save model every N rounds
    'eval_interval': 5,   # Evaluate model every N rounds
    'log_interval': 1     # Log metrics every N rounds
} 