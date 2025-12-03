"""
Linear Deep Q-Network (DQN) Implementation

This module implements a simple linear Q-network for deep reinforcement learning.
While not as powerful as convolutional architectures, the linear model serves as:
    1. A baseline for comparing more complex architectures
    2. A rapid prototyping tool for testing algorithmic changes
    3. An educational example of core DQN components

The linear Q-network learns a direct mapping from flattened state vectors to
Q-values for each action, using a single fully-connected layer.

Architecture:
    Input: Flattened state tensor (batch_size, height * width * channels)
    Output: Q-values for each action (batch_size, num_actions)

References:
    - Mnih et al. (2015). "Human-level control through deep reinforcement learning"
    - Sutton & Barto (2018). "Reinforcement Learning: An Introduction"

Author: Sahil Bhatt
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.deep_q_learning_torch import DQN
from utils.exploration import LinearExploration, LinearSchedule
from utils.test_env import EnvTest
from configs.linear_config import config


class LinearDQN(DQN):
    """
    Linear Deep Q-Network implementation.
    
    This class extends the base DQN to implement a simple linear Q-network.
    The architecture consists of a single fully-connected layer that maps
    flattened state representations directly to Q-values.
    
    While simple, this architecture can be effective for:
        - Low-dimensional state spaces
        - Environments with simple dynamics
        - Baseline comparisons and debugging
    
    The network learns the action-value function Q(s, a) which estimates
    the expected cumulative discounted reward for taking action a in state s.
    
    Attributes:
        q_network (nn.Linear): Main Q-network for action selection
        target_network (nn.Linear): Target network for stable TD targets
        optimizer (torch.optim.Adam): Optimizer for training the Q-network
    """

    def initialize_models(self) -> None:
        """
        Initialize the Q-network and target network architectures.
        
        Creates two identical linear networks:
            1. q_network: Updated every training step via backpropagation
            2. target_network: Periodically synchronized with q_network
        
        The dual-network architecture (from DQN paper) provides stability
        by using a slowly-updating target for computing TD error targets.
        """
        state_shape = self.env.state_shape()
        img_height, img_width, n_channels = state_shape
        num_actions = self.env.num_actions()
        
        # Input dimension: flattened state vector
        input_dim = img_height * img_width * n_channels
        
        # Simple linear mapping: state -> Q-values
        self.q_network = nn.Linear(input_dim, num_actions)
        self.target_network = nn.Linear(input_dim, num_actions)

    def get_q_values(self, state: torch.Tensor, network: str = "q_network") -> torch.Tensor:
        """
        Compute Q-values for all actions given a batch of states.
        
        Forward pass through the specified network to obtain Q(s, a) 
        for all actions a.
        
        Args:
            state: Batch of states with shape (batch_size, height, width, channels)
            network: Which network to use ("q_network" or "target_network")
            
        Returns:
            Q-values tensor of shape (batch_size, num_actions)
        """
        # Flatten spatial dimensions: (batch, h, w, c) -> (batch, h*w*c)
        flattened = state.view(state.size(0), -1)
        
        if network == "q_network":
            return self.q_network(flattened)
        elif network == "target_network":
            return self.target_network(flattened)
        else:
            raise ValueError(f"Unknown network: {network}")

    def update_target(self) -> None:
        """
        Synchronize target network with Q-network.
        
        Copies all parameters from q_network to target_network.
        This "hard update" is performed periodically to provide
        stable TD targets during training.
        
        The target network update frequency is a key hyperparameter:
            - Too frequent: Training instability (like no target network)
            - Too infrequent: Slow learning from stale targets
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def calc_loss(
        self,
        q_values: torch.Tensor,
        target_q_values: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the DQN loss (Mean Squared TD Error).
        
        Implements the loss function from the DQN paper:
            L = E[(r + γ * max_a' Q_target(s', a') - Q(s, a))^2]
        
        The loss measures how well the Q-network predicts the bootstrapped
        TD target computed using the target network.
        
        Args:
            q_values: Q(s, a) for all actions, shape (batch_size, num_actions)
            target_q_values: Q_target(s', a') for all actions
            actions: Actions taken, shape (batch_size,)
            rewards: Rewards received, shape (batch_size,)
            done_mask: Terminal state indicators, shape (batch_size,)
            
        Returns:
            Scalar MSE loss for the batch
        """
        gamma = self.config.gamma
        
        # Compute TD target: r + γ * max_a' Q_target(s', a')
        # For terminal states, target is just the reward (no future value)
        max_next_q = target_q_values.max(dim=1).values
        td_target = rewards + gamma * max_next_q * (~done_mask)
        
        # Get Q-values for actions actually taken
        actions = actions.long()
        q_selected = q_values.gather(1, actions.view(-1, 1)).squeeze()
        
        # MSE loss between predicted and target Q-values
        loss = F.mse_loss(q_selected, td_target)
        
        return loss

    def add_optimizer(self) -> None:
        """
        Initialize the optimizer for training.
        
        Uses Adam optimizer with default learning rate (adjusted by scheduler).
        Adam is preferred for DQN due to its adaptive learning rates
        and momentum, which help with the non-stationary targets.
        """
        self.optimizer = torch.optim.Adam(self.q_network.parameters())


def main():
    """Run training on a test environment to validate implementation."""
    # Suppress matplotlib font warnings
    logging.getLogger("matplotlib.font_manager").disabled = True
    
    # Create test environment
    env = EnvTest((5, 5, 1))
    
    # Initialize exploration schedule: epsilon 1.0 -> 0.01 over nsteps
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
    
    # Create and train model
    model = LinearDQN(env, config)
    model.run(exp_schedule, lr_schedule, run_idx=1)
    
    print("Linear DQN training completed successfully!")


if __name__ == "__main__":
    main()
