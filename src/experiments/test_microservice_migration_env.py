import numpy as np
from environment.microservice_migration_env import MicroserviceMigrationEnv

if __name__ == "__main__":
    # Create the environment
    env = MicroserviceMigrationEnv(num_nodes=3, num_services=5)

    # Reset the environment
    obs = env.reset()
    print("Initial observation (service locations):", obs)
    print("DAG structure (edges):", list(env.dag.edges))

    # Run a few random steps
    for step in range(5):
        action = env.action_space.sample()
        print(f"\nStep {step+1}")
        print("Sampled action (service_id, target_node):", action)
        next_obs, reward, done, info = env.step(action)
        print("Next observation:", next_obs)
        print("Reward:", reward)
        print("Done:", done)
        if done:
            print("Episode finished.")
            break 