"""
Test Environment for Debugging DQN Implementations

This module provides a simple, deterministic test environment for validating
DQN implementations before running expensive training on real environments.

The test environment is designed to:
    1. Provide predictable state transitions for debugging
    2. Have a small state/action space for fast iteration
    3. Mirror the MinAtar environment interface
    4. Enable unit testing of DQN components

Usage:
    >>> env = EnvTest((5, 5, 1))
    >>> state = env.reset()
    >>> reward, done = env.act(action)
    >>> current_state = env.state()

Author: Sahil Bhatt
"""

import numpy as np
from typing import Tuple


class ActionSpace:
    """Simple action space with discrete actions."""
    
    def __init__(self, n: int) -> None:
        self.n = n

    def sample(self) -> int:
        """Sample a random action."""
        return np.random.randint(0, self.n)


class ObservationSpace:
    """Observation space with random initial states."""
    
    def __init__(self, shape: Tuple[int, ...]) -> None:
        self.shape = shape
        # Create distinct states for testing transitions
        self.state_0 = np.random.randint(0, 50, shape, dtype=np.int32)
        self.state_1 = np.random.randint(100, 150, shape, dtype=np.int32)
        self.state_2 = np.random.randint(200, 250, shape, dtype=np.int32)
        self.state_3 = np.random.randint(300, 350, shape, dtype=np.int32)
        self.states = [self.state_0, self.state_1, self.state_2, self.state_3]


class EnvTest:
    """
    Test environment for validating DQN implementations.
    
    A simple environment with 4 states and 5 actions for testing:
        - Actions 0-3: Transition to corresponding state
        - Action 4: No-op (stay in current state)
    
    The environment terminates after 5 steps and provides
    deterministic rewards based on state.
    
    Attributes:
        cur_state: Current state index (0-3)
        num_iters: Number of steps taken
        
    Example:
        >>> env = EnvTest((10, 10, 4))
        >>> env.reset()
        >>> reward, done = env.act(2)  # Go to state 2
    """

    def __init__(self, shape: Tuple[int, ...] = (10, 10, 4), high: int = 255) -> None:
        """
        Initialize the test environment.
        
        Args:
            shape: Shape of state observations (height, width, channels)
            high: Maximum pixel value (for normalization)
        """
        # Fixed rewards for each state
        self.rewards = [0.1, -0.3, 0.0, -0.2]
        self.cur_state = 0
        self.num_iters = 0
        self.was_in_second = False
        
        # MinAtar-compatible interface
        self.state_shape = lambda: shape
        self.num_actions = lambda: 5
        
        self._action_space = ActionSpace(5)
        self._observation_space = ObservationSpace(shape)
        self._high = high

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.cur_state = 0
        self.num_iters = 0
        self.was_in_second = False
        return self._observation_space.states[self.cur_state]

    def act(self, action: int) -> Tuple[float, bool]:
        """
        Take an action in the environment.
        
        Args:
            action: Action index (0-4)
            
        Returns:
            reward: Reward for this step
            done: Whether episode has terminated
        """
        assert 0 <= action <= 4, f"Invalid action: {action}"
        
        self.num_iters += 1
        
        # Transition to new state if action < 4
        if action < 4:
            self.cur_state = action
            
        reward = self.rewards[self.cur_state]
        
        # Bonus/penalty logic for testing
        if self.was_in_second:
            reward *= -10
        if self.cur_state == 2:
            self.was_in_second = True
        else:
            self.was_in_second = False
            
        return reward, self.num_iters >= 5

    def state(self) -> np.ndarray:
        """Get current state observation (normalized)."""
        return self._observation_space.states[self.cur_state] / self._high

    def render(self) -> None:
        """Print current state for debugging."""
        print(f"Current state: {self.cur_state}")
