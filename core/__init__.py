"""
Core reinforcement learning algorithms.

This package contains the core Q-learning implementations:
    - q_learning: Base Q-learning class with training infrastructure
    - deep_q_learning_torch: PyTorch-based Deep Q-Network implementation

Author: Sahil Bhatt
"""

from core.q_learning import QN, Timer
from core.deep_q_learning_torch import DQN

__all__ = ["QN", "DQN", "Timer"]
