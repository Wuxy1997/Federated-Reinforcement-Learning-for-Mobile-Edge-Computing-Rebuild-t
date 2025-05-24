import torch
import numpy as np
from copy import deepcopy

class FederatedLearning:
    def __init__(self, num_clients, aggregation_rounds=10):
        self.num_clients = num_clients
        self.aggregation_rounds = aggregation_rounds
        self.global_model = None
        self.client_models = []
    
    def initialize_global_model(self, model):
        """Initialize the global model"""
        self.global_model = deepcopy(model)
    
    def initialize_client_models(self, model_template):
        """Initialize client models"""
        self.client_models = [deepcopy(model_template) for _ in range(self.num_clients)]
    
    def federated_averaging(self, client_weights, client_sizes):
        """
        Perform federated averaging of client models
        
        Args:
            client_weights: List of model state dictionaries from clients
            client_sizes: List of number of samples used by each client
        """
        total_size = sum(client_sizes)
        weights = [size / total_size for size in client_sizes]
        
        # Initialize averaged weights
        averaged_weights = {}
        for key in client_weights[0].keys():
            averaged_weights[key] = torch.zeros_like(client_weights[0][key])
        
        # Weighted average of model parameters
        for client_weight, weight in zip(client_weights, weights):
            for key in averaged_weights.keys():
                averaged_weights[key] += weight * client_weight[key]
        
        return averaged_weights
    
    def update_global_model(self, client_weights, client_sizes):
        """Update global model using federated averaging"""
        averaged_weights = self.federated_averaging(client_weights, client_sizes)
        self.global_model.load_state_dict(averaged_weights)
    
    def distribute_global_model(self):
        """Distribute global model to all clients"""
        for client_model in self.client_models:
            client_model.load_state_dict(self.global_model.state_dict())
    
    def train_round(self, client_train_functions, client_data_sizes):
        """
        Perform one round of federated learning
        
        Args:
            client_train_functions: List of training functions for each client
            client_data_sizes: List of data sizes for each client
        """
        # Train each client model
        client_weights = []
        for i, (train_fn, model) in enumerate(zip(client_train_functions, self.client_models)):
            # Train client model
            train_fn(model)
            # Get updated weights
            client_weights.append(model.state_dict())
        
        # Update global model
        self.update_global_model(client_weights, client_data_sizes)
        
        # Distribute updated global model to clients
        self.distribute_global_model()
    
    def run_federated_learning(self, client_train_functions, client_data_sizes):
        """
        Run complete federated learning process
        
        Args:
            client_train_functions: List of training functions for each client
            client_data_sizes: List of data sizes for each client
        """
        for round in range(self.aggregation_rounds):
            print(f"Federated Learning Round {round + 1}/{self.aggregation_rounds}")
            self.train_round(client_train_functions, client_data_sizes)
    
    def get_global_model(self):
        """Get the current global model"""
        return self.global_model
    
    def get_client_models(self):
        """Get all client models"""
        return self.client_models 