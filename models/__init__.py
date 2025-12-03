"""
Models package for Deep Q-Network implementations.

This package contains neural network architectures for deep reinforcement learning:

- linear_dqn: Simple fully-connected Q-network for baseline experiments
- nature_dqn: Convolutional DQN from DeepMind's Nature paper (Mnih et al., 2015)

Author: Sahil Bhatt
"""

from models.linear_dqn import LinearDQN
from models.nature_dqn import NatureDQN

__all__ = ["LinearDQN", "NatureDQN"]
