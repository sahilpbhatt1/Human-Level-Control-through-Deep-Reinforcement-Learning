"""
Experience Replay Buffer for Deep Reinforcement Learning

This module implements an experience replay buffer, a key component of the DQN
algorithm that stores transitions for training. Experience replay provides
several benefits for stable deep RL training:

    1. Breaks temporal correlations: Sequential transitions are highly correlated,
       which violates the i.i.d. assumption of stochastic gradient descent.
       Random sampling from the buffer provides uncorrelated samples.
    
    2. Improves sample efficiency: Each transition can be reused multiple times
       for training, making better use of environment interactions.
    
    3. Enables mini-batch learning: Allows training with larger batch sizes
       for more stable gradient estimates.

The buffer implements a circular/ring buffer with O(1) insertion and O(n) sampling
where n is the batch size.

References:
    - Mnih et al. (2015). "Human-level control through deep reinforcement learning"
    - Lin (1992). "Self-improving reactive agents based on reinforcement learning"

Author: Sahil Bhatt
"""

from collections import namedtuple
import random
from typing import Tuple, List

import torch


# Named tuple for storing transitions
# This provides clear, self-documenting access to transition components
Transition = namedtuple(
    "Transition", 
    ["state", "next_state", "action", "reward", "is_terminal"]
)


class ReplayBuffer:
    """
    Fixed-size circular buffer for storing and sampling experience transitions.
    
    Stores transitions as (s, s', a, r, done) tuples and provides uniform
    random sampling for training. When the buffer is full, oldest transitions
    are overwritten following a FIFO policy.
    
    This implementation is optimized for MinAtar environments and stores
    transitions as PyTorch tensors for efficient GPU transfer during training.
    
    Attributes:
        buffer_size: Maximum number of transitions to store
        location: Current write position in circular buffer
        buffer: List of stored Transition namedtuples
    
    Example:
        >>> buffer = ReplayBuffer(buffer_size=100000)
        >>> buffer.add(state, next_state, action, reward, done)
        >>> states, next_states, actions, rewards, dones = buffer.sample(32)
    """
    
    def __init__(self, buffer_size: int) -> None:
        """
        Initialize the replay buffer.
        
        Args:
            buffer_size: Maximum capacity of the buffer
        """
        self.buffer_size = buffer_size
        self.location = 0  # Next write position
        self.buffer: List[Transition] = []

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return len(self.buffer)

    def add(
        self, 
        state: torch.Tensor, 
        next_state: torch.Tensor, 
        action: torch.Tensor, 
        reward: torch.Tensor, 
        is_terminal: torch.Tensor
    ) -> None:
        """
        Add a transition to the buffer.
        
        If the buffer is not full, appends the transition.
        If full, overwrites the oldest transition (circular buffer).
        
        Args:
            state: Current state observation
            next_state: Next state observation
            action: Action taken
            reward: Reward received
            is_terminal: Whether next_state is terminal
        """
        transition = Transition(state, next_state, action, reward, is_terminal)
        
        if len(self.buffer) < self.buffer_size:
            # Buffer not full, append
            self.buffer.append(transition)
        else:
            # Buffer full, overwrite oldest
            self.buffer[self.location] = transition

        # Update write position (circular)
        self.location = (self.location + 1) % self.buffer_size

    def sample(
        self, 
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a random batch of transitions for training.
        
        Performs uniform random sampling without replacement from the buffer.
        Returns tensors ready for neural network training.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of tensors:
                - states: (batch_size, *state_shape)
                - next_states: (batch_size, *state_shape)
                - actions: (batch_size,)
                - rewards: (batch_size,)
                - dones: (batch_size,) terminal flags
                
        Raises:
            ValueError: If batch_size > len(buffer)
        """
        # Sample random transitions
        samples = random.sample(self.buffer, batch_size)
        
        # Unzip into separate lists and stack into tensors
        batch = Transition(*zip(*samples))
        
        states = torch.cat(batch.state)
        next_states = torch.cat(batch.next_state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward).flatten()
        dones = torch.cat(batch.is_terminal).flatten()
        
        return states, next_states, actions, rewards, dones

    def can_sample(self, batch_size: int) -> bool:
        """
        Check if buffer has enough samples for a batch.
        
        Args:
            batch_size: Desired batch size
            
        Returns:
            True if buffer contains at least batch_size transitions
        """
        return len(self.buffer) >= batch_size