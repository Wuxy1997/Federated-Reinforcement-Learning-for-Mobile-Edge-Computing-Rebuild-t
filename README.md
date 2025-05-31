# Federated Reinforcement Learning for Mobile Edge Computing

This project implements a robust federated reinforcement learning framework for mobile edge computing (MEC) environments, with a focus on microservice migration optimization.

## Project Structure

```
├── src/
│   ├── agents/                 # RL agent implementations
│   ├── environment/            # Environment implementations
│   ├── federated/              # Federated learning implementations
│   ├── robust/                 # Robust aggregation methods
│   ├── utils/                  # Utility functions
│   ├── configs/                # Configuration files
│   ├── experiments/            # Experiment scripts
│   └── main.py                 # Main training script
├── requirements.txt            # Project dependencies
├── webapp.py                   # Web interface for visualization
├── templates/                  # Web frontend templates
├── static/results/             # Output directory for results
├── FRL_code_explanation_en.md  # English code explanation
└── README.md                   # Project documentation
```

## Features

- **Robust Federated Learning**: Implements multiple robust aggregation methods (Krum, Median, Trimmed Mean) to defend against model poisoning attacks. These methods ensure that the global model remains resilient even when some clients are compromised.
- **Multiple Environments**: Supports both a basic MEC environment and a microservice migration environment with DAG support, allowing for flexible experimentation and optimization.
- **Model Poisoning Simulation**: Simulates attacks such as random noise and sign flip, enabling researchers to evaluate the robustness of the federated learning system under adversarial conditions.
- **Automatic Visualization**: Generates comprehensive visualizations including training curves, reward distributions, client comparisons, service migration heatmaps, aggregation method comparisons, and attack impact analysis.
- **Web Interface**: Provides a user-friendly web interface to configure parameters, run experiments, and browse/download results, making it accessible for both researchers and practitioners.

## Principles

- **Federated Learning**: The system employs federated learning to train models across multiple devices without sharing raw data, thus preserving privacy and reducing communication overhead.
- **Reinforcement Learning**: Utilizes reinforcement learning algorithms to optimize decision-making processes in dynamic environments, such as task offloading and microservice migration.
- **Robust Aggregation**: Implements robust aggregation techniques to mitigate the impact of adversarial attacks, ensuring the integrity and reliability of the global model.
- **Visualization and Analysis**: Emphasizes the importance of visualization and analysis tools to interpret results and gain insights into the performance of the federated learning system.

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

## Quick Start

### Web Interface

1. Start the web server:
   ```bash
   python webapp.py
   ```

2. Open your browser and navigate to [http://localhost:5000](http://localhost:5000).

3. Configure your experiment parameters on the web page, click "Run," and view the automatically generated visualizations and results. You can download images and CSV data directly from the interface.

## Results and Visualization

- The training process automatically generates the following results (default in `static/results/`):
  - `training_results.png`: Global average reward curve
  - `training_rewards.csv`: Detailed reward data
  - `client_rewards.png`: Client-wise reward comparison
  - `reward_distribution.png`: Reward distribution histogram
  - `aggregation_comparison.png`: Aggregation method comparison
  - `attack_impact.png`: Attack impact analysis
  - `client_evolution.png`: Client performance evolution
  - Additional visualizations such as service migration heatmaps and DAG structure (if migration environment is enabled)

## Testing and Extension

- **Environment Testing**: Run `src/experiments/test_microservice_migration_env.py` to verify the microservice migration environment and DAG dependencies:
  ```bash
  python src/experiments/test_microservice_migration_env.py
  ```

- For detailed code structure and extension suggestions, refer to `FRL_code_explanation_en.md`.

## Acknowledgments

- This project is an independent implementation inspired by the literature on Federated Reinforcement Learning for Mobile Edge Computing.
- The code and documentation are original and suitable for academic research and engineering applications.

## License

MIT License, see LICENSE file for details.