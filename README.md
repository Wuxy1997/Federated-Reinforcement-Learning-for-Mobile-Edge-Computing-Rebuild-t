# Federated Reinforcement Learning for Mobile Edge Computing
This project implements a robust federated reinforcement learning framework for mobile edge computing (MEC) environments, with a focus on microservice migration optimization.

## Project Structure

```
├── src/
│   ├── agents/                 # RL agent implementations
│   │   ├── dqn_agent.py       # DQN agent implementation
│   │   └── ddpg_agent.py      # DDPG agent implementation
│   ├── environment/           # Environment implementations
│   │   ├── mec_env.py         # Basic MEC environment
│   │   └── microservice_migration_env.py  # Microservice migration environment
│   ├── federated/             # Federated learning implementations
│   │   ├── federated_learning.py          # Standard federated learning
│   │   └── robust_federated_learning.py   # Robust federated learning
│   ├── robust/               # Robust aggregation methods
│   │   ├── krum.py           # Krum aggregation
│   │   ├── median.py         # Median aggregation
│   │   └── trimmed_mean.py   # Trimmed mean aggregation
│   ├── utils/                # Utility functions
│   │   ├── visualization.py  # Visualization tools
│   │   └── metrics.py        # Performance metrics
│   └── main.py              # Main training script
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Features

- **Robust Federated Learning**: Implements multiple robust aggregation methods:
  - Krum
  - Median
  - Trimmed Mean
- **Model Poisoning Defense**: Built-in defense against model poisoning attacks
- **Multiple Environments**:
  - Basic MEC environment
  - Microservice migration environment with DAG support
- **Visualization Tools**:
  - Training progress visualization
  - DAG structure visualization
  - Migration animation
  - Performance metrics plotting

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Run the main training script with default settings:
```bash
python src/main.py
```

### Configuration

The main training parameters can be configured in `src/main.py`:

```python
# Federated Learning Settings
USE_ROBUST_FL = True  # Use robust federated learning
AGGREGATION_METHOD = 'krum'  # Aggregation method
NUM_ADV = 1  # Number of adversarial clients

# Attack Settings
ENABLE_ATTACK = True  # Enable model poisoning attack
ATTACK_METHOD = 'random_noise'  # Attack method
ATTACK_EPSILON = 10.0  # Attack strength

# Environment Settings
USE_MIGRATION_ENV = False  # Use microservice migration environment
NUM_EDGE_NODES = 3
NUM_MOBILE_DEVICES = 5
NUM_SERVICES = 5

# Training Settings
EPISODES_PER_ROUND = 100
NUM_ROUNDS = 10
```

### Visualization

The training process generates several visualization files:
- `training_results.png`: Global average reward curve
- `training_rewards.csv`: Detailed reward data
- `initial_dag_locations.png`: Initial DAG structure (when using migration environment)
- `global_model.pth`: Trained global model

## Robust Aggregation Methods

### Krum
- Selects the model update that is closest to its neighbors
- Effective against Byzantine attacks
- Parameters:
  - `num_adv`: Number of adversarial clients to defend against

### Median
- Takes the median of all model updates for each parameter
- Robust against outliers
- No additional parameters required

### Trimmed Mean
- Removes a percentage of extreme values before averaging
- Parameters:
  - `trim_ratio`: Percentage of values to trim from each end

## Model Poisoning Attacks

The framework includes simulation of model poisoning attacks:
- Random Noise: Adds random noise to model parameters
- Sign Flip: Flips the signs of model parameters
- Attack strength can be controlled via `ATTACK_EPSILON`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project is mainly for me appling RA of Federated Reinforcement Learning for Mobile Edge Computing area
- Special thanks to all contributors and researchers in the field
- This readme file is write by AI(laugh)