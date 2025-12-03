# Deep Q-Network (DQN) Implementation for Atari Games

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains a **production-quality implementation** of Deep Q-Network (DQN), reproducing key results from DeepMind's groundbreaking paper ["Human-level control through deep reinforcement learning"](https://www.nature.com/articles/nature14236) (Mnih et al., 2015). The implementation relies on **reinforcement learning**, **sequential decision-making**, and **deep learning** techniques using PyTorch.

### Key Features

- **Complete DQN Architecture**: Implementation of both linear and convolutional (Nature DQN) Q-networks
- **Experience Replay**: Efficient replay buffer for breaking temporal correlations in training data
- **Target Network**: Separate target network for stable Q-value estimation
- **Epsilon-Greedy Exploration**: Linear annealing schedule for exploration-exploitation tradeoff
- **TensorBoard Integration**: Real-time visualization of training metrics
- **MinAtar Environment**: Lightweight Atari-like environment for rapid prototyping and experimentation

## Table of Contents

- [Theoretical Background](#theoretical-background)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Details](#technical-details)
- [References](#references)

## Theoretical Background

### Deep Q-Learning

Deep Q-Learning combines Q-learning with deep neural networks to learn optimal policies in high-dimensional state spaces. The agent learns a Q-function $Q(s, a; \theta)$ parameterized by neural network weights $\theta$ that estimates the expected cumulative reward for taking action $a$ in state $s$.

#### Bellman Optimality Equation

The optimal Q-function satisfies the Bellman equation:

$$Q^{\ast}(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^{\ast}(s', a') \mid s, a\right]$$

#### DQN Loss Function

The network is trained by minimizing the temporal difference (TD) error:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}}\left[\left(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]$$

where $\theta^-$ represents the parameters of the target network and $\mathcal{D}$ is the replay buffer.

### Key Algorithmic Innovations

1. **Experience Replay**: Stores transitions $(s, a, r, s')$ in a buffer and samples mini-batches uniformly, breaking correlations between consecutive samples
2. **Target Network**: Uses a separate, periodically updated network for computing TD targets, improving training stability
3. **Epsilon-Greedy Exploration**: Balances exploration and exploitation through annealing exploration rate

## Project Architecture

```
├── core/                           # Core RL algorithms
│   ├── q_learning.py              # Base Q-learning agent with training loop
│   └── deep_q_learning_torch.py   # PyTorch DQN implementation
├── models/
│   ├── linear_dqn.py              # Linear Q-network architecture
│   └── nature_dqn.py              # Convolutional (Nature) DQN architecture
├── configs/                        # Hyperparameter configurations
│   ├── linear_config.py           # Config for linear DQN experiments
│   └── nature_config.py           # Config for Nature DQN experiments
├── utils/                          # Utility modules
│   ├── replay_buffer.py           # Experience replay implementation
│   ├── exploration.py             # Exploration strategies (ε-greedy)
│   └── visualization.py           # Plotting and logging utilities
├── train_linear.py                 # Training script for linear DQN
├── train_nature.py                 # Training script for Nature DQN
└── requirements.txt               # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/deep-q-network-atari.git
   cd deep-q-network-atari
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install MinAtar environment**
   ```bash
   git clone https://github.com/kenjyoung/MinAtar.git
   cd MinAtar && pip install . && cd ..
   ```

## Usage

### Training

**Linear DQN** (for quick experimentation):
```bash
python train_linear.py
```

**Nature DQN** (full convolutional architecture):
```bash
python train_nature.py
```

### Monitoring Training

Launch TensorBoard to visualize training progress:
```bash
tensorboard --logdir=results/
```

### Configuration

Hyperparameters can be modified in the config files located in `configs/`. Key parameters include:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `nsteps_train` | Total training steps | 1,000,000 |
| `batch_size` | Mini-batch size for training | 32 |
| `buffer_size` | Replay buffer capacity | 100,000 |
| `gamma` | Discount factor | 0.99 |
| `eps_begin` | Initial exploration rate | 1.0 |
| `eps_end` | Final exploration rate | 0.1 |
| `target_update_freq` | Target network update frequency | 1,000 |
| `learning_rate` | Adam optimizer learning rate | 0.00025 |

## Results

### MinAtar Breakout Performance

The Nature DQN architecture achieves strong performance on MinAtar Breakout:

- **Average Score**: Achieves competitive scores after 1M training steps
- **Learning Curve**: Shows consistent improvement with proper hyperparameter tuning
- **Convergence**: Stable learning with target network updates every 1,000 steps

Training produces evaluation scores saved in `results/` along with TensorBoard logs for detailed analysis.

## Technical Details

### Neural Network Architectures

#### Linear DQN
- Single fully-connected layer mapping flattened state to Q-values
- Suitable for simple environments and baseline comparisons

#### Nature DQN (Convolutional)
```
Input (10×10×4) → Conv2D(16, 3×3) → ReLU → Flatten → FC(128) → ReLU → FC(num_actions)
```

### Exploration Strategy

Implements **linear epsilon decay**:
- Epsilon starts at 1.0 (fully random actions)
- Linearly decays to 0.1 over first 100,000 steps
- Ensures sufficient exploration during early training while exploiting learned policy later

### Experience Replay

- **Buffer Size**: 100,000 transitions
- **Sampling**: Uniform random sampling for training batches
- **Memory Efficiency**: Stores transitions as PyTorch tensors on CPU
 

## References

1. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.
2. Mnih, V., et al. (2013). "Playing Atari with Deep Reinforcement Learning." *arXiv preprint arXiv:1312.5602*.
3. Young, K., & Tian, T. (2019). "MinAtar: An Atari-Inspired Testbed for Thorough and Reproducible Reinforcement Learning Experiments." *arXiv preprint arXiv:1903.03176*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This implementation was developed as part of advanced coursework in reinforcement learning, demonstrating practical skills in sequential decision-making algorithms relevant to forecasting, optimization, and AI-driven solutions.*


Training tips: 
(1) The starter code writes summaries of a bunch of useful variables that can help you monitor the training process.
You can monitor your training with Tensorboard by doing, on Azure

```
tensorboard --logdir=results
```

and then connect to `ip-of-you-machine:6006`


(2) You can use ‘screen’ to manage windows on VM and to retrieve running programs. 
Before training DQN on Atari games, run 

```
screen 
```
then run 

```
python q6_train_atari_nature.py
```
By using Screen, programs continue to run when their window is currently not visible and even when the whole screen session is detached 
from the users terminal. 

To detach from your window, simply press the following sequence of buttons

```
ctrl-a d
```
This is done by pressing control-a first, releasing it, and press d


To retrieve your running program on VM, simply type

```
screen -r
```
which will recover the detached window.   



**Credits**
Assignment code written by Guillaume Genthial and Shuhui Qu.
Assignment code updated by Jian Vora and Max Sobol Mark
