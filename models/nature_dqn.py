"""
Nature DQN: Convolutional Deep Q-Network Implementation

This module implements the convolutional neural network architecture from DeepMind's
landmark Nature paper "Human-level control through deep reinforcement learning" 
(Mnih et al., 2015). This architecture achieved human-level performance on many
Atari 2600 games, demonstrating that deep reinforcement learning can solve
complex sequential decision-making problems from raw pixel inputs.

Architecture Overview:
    The network processes visual observations through convolutional layers
    that extract spatial features, followed by fully-connected layers that
    learn the state-action value function Q(s, a).

    Input: State tensor (batch_size, height, width, channels)
    Conv2D: 16 filters, 3x3 kernel, stride 1 -> ReLU
    Flatten -> FC(128) -> ReLU -> FC(num_actions)
    Output: Q-values for each action

Key Design Principles:
    1. Convolutional layers share weights across spatial locations,
       making the representation translation-invariant
    2. ReLU activations provide non-linearity for learning complex mappings
    3. The architecture is adapted for MinAtar's smaller state space
       (10x10 instead of Atari's 84x84)

References:
    - Mnih et al. (2015). "Human-level control through deep reinforcement learning"
    - Young & Tian (2019). "MinAtar: An Atari-Inspired Testbed"

Author: Sahil Bhatt
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.exploration import LinearExploration, LinearSchedule
from utils.test_env import EnvTest
from models.linear_dqn import LinearDQN
from configs.nature_config import config


class NatureDQN(LinearDQN):
    """
    Convolutional Deep Q-Network (Nature DQN) implementation.
    
    This class implements the CNN architecture from the DQN Nature paper,
    adapted for the MinAtar environment. The convolutional architecture
    enables learning directly from pixel inputs without hand-crafted features.
    
    The key insight of DQN is that a deep neural network can approximate
    the optimal action-value function Q*(s, a) when combined with:
        1. Experience replay: Breaks temporal correlations in training data
        2. Target network: Provides stable TD targets during training
        3. Reward clipping: Normalizes gradient magnitudes across games
    
    Network Architecture:
        - Conv2D(in_channels -> 16, kernel=3x3, stride=1) + ReLU
        - Flatten
        - Linear(conv_output_size -> 128) + ReLU  
        - Linear(128 -> num_actions)
    
    The convolutional layer extracts spatial features from the game state,
    while the fully-connected layers learn to map these features to Q-values.
    
    Attributes:
        q_network (nn.Sequential): Main convolutional Q-network
        target_network (nn.Sequential): Target network for stable TD targets
    """

    def initialize_models(self) -> None:
        """
        Initialize the convolutional Q-network and target network.
        
        Creates two identical CNN architectures:
            1. q_network: Updated every training step
            2. target_network: Periodically synchronized (every target_update_freq steps)
        
        Architecture Details:
            - Conv2D: Extracts 16 feature maps using 3x3 filters
            - The output size is computed based on input dimensions
            - Two FC layers map features to action values
        """
        state_shape = self.env.state_shape()
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.num_actions()
        
        # Compute size after convolution: output = (input - kernel_size) / stride + 1
        conv_out_height = (img_height - 3) // 1 + 1  # (H - K) / S + 1
        conv_out_width = (img_width - 3) // 1 + 1
        conv_out_features = 16 * conv_out_height * conv_out_width
        
        # Build the Q-network architecture
        self.q_network = nn.Sequential(
            # Convolutional layer: extract spatial features
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            
            # Flatten for fully-connected layers
            nn.Flatten(),
            
            # First FC layer: learn feature combinations
            nn.Linear(in_features=conv_out_features, out_features=128),
            nn.ReLU(),
            
            # Output layer: Q-value for each action
            nn.Linear(in_features=128, out_features=num_actions)
        )
        
        # Target network: identical architecture, separate parameters
        self.target_network = nn.Sequential(
            nn.Conv2d(
                in_channels=n_channels,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=conv_out_features, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_actions)
        )

    def get_q_values(self, state: torch.Tensor, network: str) -> torch.Tensor:
        """
        Compute Q-values using the convolutional network.
        
        The state tensor must be permuted from (B, H, W, C) to (B, C, H, W)
        format expected by PyTorch's Conv2d layers.
        
        Args:
            state: Batch of states, shape (batch_size, height, width, channels)
            network: Network to use ("q_network" or "target_network")
            
        Returns:
            Q-values tensor of shape (batch_size, num_actions)
            
        Note:
            PyTorch convolutions expect channels-first format (N, C, H, W),
            but our states are stored in channels-last format (N, H, W, C)
            for compatibility with OpenAI Gym conventions.
        """
        # Convert from channels-last to channels-first format
        # (batch, height, width, channels) -> (batch, channels, height, width)
        state = state.permute(0, 3, 1, 2)
        
        if network == "q_network":
            return self.q_network(state)
        elif network == "target_network":
            return self.target_network(state)
        else:
            raise ValueError(f"Unknown network: {network}")


def main():
    """Run training on a test environment to validate the CNN implementation."""
    # Suppress matplotlib font warnings
    logging.getLogger("matplotlib.font_manager").disabled = True
    
    # Create test environment with 6 channels (MinAtar-like)
    env = EnvTest((8, 8, 6))
    
    # Initialize exploration schedule: epsilon 1.0 -> 0.01
    exp_schedule = LinearExploration(
        env,
        config.eps_begin,
        config.eps_end,
        config.eps_nsteps
    )
    
    # Initialize learning rate schedule
    lr_schedule = LinearSchedule(
        config.lr_begin,
        config.lr_end,
        config.lr_nsteps
    )
    
    # Create and train the Nature DQN model
    model = NatureDQN(env, config)
    model.run(exp_schedule, lr_schedule, run_idx=1)
    
    print("Nature DQN training completed successfully!")


if __name__ == "__main__":
    main()
