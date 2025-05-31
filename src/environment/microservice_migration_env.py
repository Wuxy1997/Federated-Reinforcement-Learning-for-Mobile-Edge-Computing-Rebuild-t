import numpy as np
import gym
from gym import spaces
import networkx as nx

class MicroserviceMigrationEnv(gym.Env):
    """
    Microservice Migration Environment with DAG Task Support
    """
    def __init__(self, num_nodes=3, num_services=5):
        super(MicroserviceMigrationEnv, self).__init__()
        self.num_nodes = num_nodes
        self.num_services = num_services

        # Generate a random DAG for microservice dependencies
        self.dag = self._generate_random_dag(num_services)
        self.service_list = list(self.dag.nodes)

        # State: [service_locations, node_resources, ...]
        # For simplicity, service_locations: [node_id for each service]
        self.observation_space = spaces.Box(
            low=0, high=num_nodes-1, shape=(num_services,), dtype=np.int32
        )
        # Action: (service_id, target_node_id) or "no migration"
        self.action_space = spaces.MultiDiscrete([num_services, num_nodes + 1])  # +1 for "no migration"

        self.reset()

    def _generate_random_dag(self, num_services):
        """Generate a random DAG using networkx"""
        dag = nx.gn_graph(num_services, seed=None, create_using=nx.DiGraph)
        dag = nx.relabel_nodes(dag, lambda x: int(x))
        return dag

    def reset(self):
        """Reset the environment to the initial state"""
        self.service_locations = np.random.randint(0, self.num_nodes, size=self.num_services)
        self.node_resources = np.random.uniform(0.5, 1.0, size=(self.num_nodes,))
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        """Take an action: migrate a microservice or do nothing"""
        service_id, target_node = action
        done = False

        # If target_node == self.num_nodes, means "no migration"
        if target_node < self.num_nodes:
            self.service_locations[service_id] = target_node

        # Calculate reward: negative of total DAG completion time + migration penalty
        total_delay = self._calculate_dag_completion_time()
        migration_penalty = 0.1  # You can design a more complex penalty
        reward = -total_delay - migration_penalty
        reward = reward + 10
        reward = np.clip(reward, 0, 20)
        reward = reward / 20  # Normalize to [0, 1]

        self.current_step += 1
        if self.current_step >= 20:
            done = True

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """Return the current observation (service locations)"""
        return self.service_locations.copy()

    def _calculate_dag_completion_time(self):
        """Calculate the total completion time of the DAG task (simple version)"""
        total = 0
        for service in self.service_list:
            node = self.service_locations[service]
            total += 1.0 / self.node_resources[node]
        # You can add dependency-aware scheduling here
        return total 