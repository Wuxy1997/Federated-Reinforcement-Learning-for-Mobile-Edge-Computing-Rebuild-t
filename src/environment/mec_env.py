import numpy as np
import gym
from gym import spaces

class MECEnvironment(gym.Env):
    """
    Mobile Edge Computing Environment
    Simulates a MEC system with multiple edge nodes and mobile devices
    """
    def __init__(self, num_edge_nodes=3, num_mobile_devices=5):
        super(MECEnvironment, self).__init__()
        
        self.num_edge_nodes = num_edge_nodes
        self.num_mobile_devices = num_mobile_devices
        self.current_device = 0  # Initialize current_device
        
        # Define action space (which edge node to offload to)
        self.action_space = spaces.Discrete(num_edge_nodes + 1)  # +1 for local execution
        
        # Define observation space
        # [device_computation_capacity, edge_node_computation_capacity, 
        #  task_size, task_deadline, channel_state]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0
        self.current_device = 0  # Reset current_device
        self.device_computation = np.random.uniform(0.2, 1.0, self.num_mobile_devices)
        self.edge_computation = np.random.uniform(0.5, 1.0, self.num_edge_nodes)
        self.task_size = np.random.uniform(0.1, 1.0, self.num_mobile_devices)
        self.task_deadline = np.random.uniform(0.5, 1.0, self.num_mobile_devices)
        self.channel_state = np.random.uniform(0.1, 1.0, (self.num_mobile_devices, self.num_edge_nodes))
        
        return self._get_observation()
    
    def step(self, action):
        """
        Execute one time step within the environment
        
        Args:
            action: Integer representing the edge node to offload to (0 for local execution)
            
        Returns:
            observation: Current state
            reward: Reward for the action
            done: Whether the episode is finished
            info: Additional information
        """
        self.current_step += 1
        
        # Calculate execution time and energy consumption
        execution_time, energy_consumption = self._calculate_metrics(action)
        
        # Calculate reward (negative of total cost)
        reward = -self._calculate_cost(execution_time, energy_consumption)
        reward = reward + 10
        reward = np.clip(reward, 0, 20)
        reward = reward / 20  # Normalize to [0, 1]
        
        # Check if deadline is met
        done = execution_time > self.task_deadline[self.current_device]
        
        # Update state
        self._update_state()
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        """Get current state observation"""
        return np.array([
            self.device_computation[self.current_device],
            np.mean(self.edge_computation),
            self.task_size[self.current_device],
            self.task_deadline[self.current_device],
            np.mean(self.channel_state[self.current_device])
        ])
    
    def _calculate_metrics(self, action):
        """Calculate execution time and energy consumption"""
        if action == 0:  # Local execution
            execution_time = self.task_size[self.current_device] / self.device_computation[self.current_device]
            energy_consumption = execution_time * self.device_computation[self.current_device]
        else:  # Edge execution
            edge_node = action - 1
            transmission_time = self.task_size[self.current_device] / self.channel_state[self.current_device, edge_node]
            execution_time = transmission_time + (self.task_size[self.current_device] / self.edge_computation[edge_node])
            energy_consumption = transmission_time * self.device_computation[self.current_device]
        
        return execution_time, energy_consumption
    
    def _calculate_cost(self, execution_time, energy_consumption):
        """Calculate total cost (weighted sum of time and energy)"""
        time_weight = 0.5
        energy_weight = 0.5
        return time_weight * execution_time + energy_weight * energy_consumption
    
    def _update_state(self):
        """Update environment state for next step"""
        self.current_device = (self.current_device + 1) % self.num_mobile_devices
        # Add some randomness to the environment
        self.device_computation *= np.random.uniform(0.95, 1.05, self.num_mobile_devices)
        self.edge_computation *= np.random.uniform(0.95, 1.05, self.num_edge_nodes)
        self.channel_state *= np.random.uniform(0.9, 1.1, (self.num_mobile_devices, self.num_edge_nodes)) 