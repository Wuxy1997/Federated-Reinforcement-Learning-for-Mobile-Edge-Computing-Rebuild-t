# Federated Reinforcement Learning (FRL) Code Explanation

This document provides a detailed explanation of the overall structure, module functions, and collaboration of the Federated Reinforcement Learning (FRL) code in this project. It is suitable for project documentation and developer reference.

---

## 1. Overall Workflow

This code implements a **Federated Reinforcement Learning system** for task offloading decisions in Mobile Edge Computing (MEC) scenarios. The main workflow is as follows:

1. **Environment Initialization**: Create a simulated MEC environment (`MECEnvironment`).
2. **Agent Initialization**: Create a DQN agent (`DQNAgent`) for each mobile device.
3. **Federated Learning Framework**: Use the `FederatedLearning` class to manage the global model and aggregate client models.
4. **Training Loop**: In each federated learning round, each agent trains locally, uploads model parameters, performs federated averaging, updates the global model, and then distributes it back to all agents.
5. **Result Visualization**: Record the average reward for each round and plot the training curve at the end.

---

## 2. Main Module Details

### 2.1 Environment Simulator (`src/environment/mec_env.py`)

- **MECEnvironment** inherits from OpenAI Gym's `Env` class and simulates an MEC environment with multiple edge nodes and mobile devices.
- **State space**: Includes device computation power, edge node computation power, task size, task deadline, channel state, etc.
- **Action space**: Each action represents offloading the task to a specific edge node or executing it locally.
- **Reward function**: Negative total cost (weighted sum of delay and energy consumption); higher reward means better performance.
- **step/reset**: Each step calculates delay and energy based on the action, updates the state, and returns the new state and reward.

### 2.2 DQN Agent (`src/agents/dqn_agent.py`)

- **DQNAgent** implements the Deep Q-Network (DQN) algorithm.
- Includes:
  - Q-network and target network
  - Experience replay buffer
  - Epsilon-greedy policy
  - Training (replay) and parameter update methods
- Each agent interacts with the environment and learns independently.

### 2.3 Federated Learning Framework (`src/federated/federated_learning.py`)

- **FederatedLearning** manages the global model and all client models.
- **federated_averaging**: Aggregates uploaded model parameters from clients by weighted average (based on sample size) to obtain a new global model.
- **update_global_model/distribute_global_model**: Aggregate and distribute the model.
- Supports multiple rounds of federated aggregation.

### 2.4 Main Script (`src/main.py`)

- Reads parameters and initializes the environment, agents, and federated learning framework.
- **Training loop**:
  1. In each round, each agent trains locally for several episodes.
  2. Collects all agents' model parameters, performs federated averaging, and updates the global model.
  3. Distributes the global model to all agents for the next round.
  4. Records the average reward for each round.
- After training, plots the reward curve and saves the final model.

---

## 3. Module Collaboration Flow (Simplified)

```
┌─────────────┐
│  configs/   │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ main.py     │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ MECEnvironment │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ DQNAgent    │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ FederatedLearning │
└─────────────┘
```

---

## 4. Key Points

- **Each agent trains locally**, only uploading model parameters (not raw data), thus protecting data privacy.
- **Federated averaging** enables knowledge sharing among agents, improving global performance.
- **Reward curve** reflects the overall system performance improvement process.

---

## 5. Robust Aggregation & Model Poisoning Attack Simulation

### Robust Aggregation
- The project supports robust aggregation methods (Krum, Median, Trimmed Mean) to defend against model poisoning attacks.
- You can select the aggregation method and configure its parameters in `src/main.py`.

### Model Poisoning Attack Simulation
- You can enable or disable model poisoning attacks via the `ENABLE_ATTACK` parameter in `src/main.py`.
- Attack methods such as `random_noise` and `sign_flip` are supported, and the attack strength can be set via `ATTACK_EPSILON`.
- The system will print which clients are poisoned in each round if attack is enabled.
- This allows you to compare the performance of standard and robust aggregation under adversarial conditions, providing a strong basis for robustness evaluation and research.

---

## 6. Possible Extensions

- Use more advanced RL algorithms (e.g., PPO, A3C, etc.)
- Increase environment complexity (e.g., dynamic tasks, heterogeneous nodes, etc.)
- Implement asynchronous federated learning, noisy local updates, and other more realistic federated scenarios

---

For more details on specific modules, parameter tuning, or extensions, please refer to this document or contact the developers! 