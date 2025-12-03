"""
Exploration Strategies for Deep Reinforcement Learning

This module implements exploration strategies for balancing the exploration-exploitation
tradeoff in reinforcement learning. The epsilon-greedy strategy with linear annealing
is implemented, which is a fundamental technique used in DQN and many other RL algorithms.

The exploration rate (epsilon) determines the probability of taking a random action
versus exploiting the current best known action. Linear annealing gradually reduces
exploration as the agent learns better policies.

References:
    - Mnih et al. (2015). "Human-level control through deep reinforcement learning"
    - Sutton & Barto (2018). "Reinforcement Learning: An Introduction"

Author: Sahil Bhatt
"""

import random
from typing import Optional

import numpy as np


class LinearSchedule:
    """
    Linear annealing schedule for hyperparameter decay.
    
    Implements linear interpolation between an initial value and a final value
    over a specified number of timesteps. Commonly used for learning rate 
    scheduling and exploration rate decay in reinforcement learning.
    
    The schedule follows:
        value(t) = begin + t/nsteps * (end - begin)    for t <= nsteps
        value(t) = end                                  for t > nsteps
    
    Attributes:
        epsilon: Current value of the scheduled parameter
        eps_begin: Initial value at t=0
        eps_end: Final value at t=nsteps
        nsteps: Number of steps for linear interpolation
    
    Example:
        >>> schedule = LinearSchedule(eps_begin=1.0, eps_end=0.1, nsteps=10000)
        >>> schedule.update(t=5000)
        >>> print(schedule.epsilon)  # 0.55 (midpoint between 1.0 and 0.1)
    """
    
    def __init__(self, eps_begin: float, eps_end: float, nsteps: int) -> None:
        """
        Initialize the linear schedule.
        
        Args:
            eps_begin: Initial value of the parameter
            eps_end: Final value after annealing completes
            nsteps: Number of timesteps over which to anneal
        """
        self.epsilon = eps_begin
        self.eps_begin = eps_begin
        self.eps_end = eps_end
        self.nsteps = nsteps

    def update(self, t: int) -> None:
        """
        Update the scheduled parameter based on current timestep.
        
        Computes the linearly interpolated value for timestep t.
        For t > nsteps, the value remains at eps_end.
        
        Args:
            t: Current timestep (frame number)
        """
        if t > self.nsteps:
            self.epsilon = self.eps_end
        else:
            # Linear interpolation: epsilon = begin + t/nsteps * (end - begin)
            progress = t / self.nsteps
            self.epsilon = self.eps_begin + progress * (self.eps_end - self.eps_begin)


class LinearExploration(LinearSchedule):
    """
    Epsilon-greedy exploration with linear annealing.
    
    This class implements the exploration strategy used in Deep Q-Networks (DQN).
    It combines epsilon-greedy action selection with a linear decay schedule
    for the exploration rate.
    
    The strategy works as follows:
        - With probability epsilon: take a random action (explore)
        - With probability 1-epsilon: take the best action (exploit)
    
    Epsilon decays linearly from eps_begin to eps_end over nsteps timesteps,
    ensuring high exploration early in training when the Q-network is unreliable,
    and low exploration later when the learned policy is more trustworthy.
    
    This exploration-exploitation tradeoff is critical for:
        1. Discovering diverse experiences to fill the replay buffer
        2. Avoiding local optima in the policy space
        3. Ensuring convergence to near-optimal policies
    
    Attributes:
        env: The environment (provides num_actions() method)
        epsilon: Current exploration rate (inherited from LinearSchedule)
    
    Example:
        >>> env = MinAtarEnvironment("breakout")
        >>> exploration = LinearExploration(env, eps_begin=1.0, eps_end=0.1, nsteps=100000)
        >>> best_action = q_network.get_best_action(state)
        >>> action = exploration.get_action(best_action)  # May be random with prob epsilon
    """
    
    def __init__(self, env, eps_begin: float, eps_end: float, nsteps: int) -> None:
        """
        Initialize epsilon-greedy exploration with linear annealing.
        
        Args:
            env: Environment instance with num_actions() method
            eps_begin: Initial exploration rate (typically 1.0 for fully random)
            eps_end: Final exploration rate (typically 0.1 or 0.01)
            nsteps: Number of timesteps for epsilon decay
        """
        self.env = env
        super().__init__(eps_begin, eps_end, nsteps)

    def get_action(self, best_action: int) -> int:
        """
        Select an action using epsilon-greedy strategy.
        
        Returns a random action with probability epsilon (exploration),
        or the provided best action with probability 1-epsilon (exploitation).
        
        Args:
            best_action: The greedy action from the Q-network
            
        Returns:
            Selected action (either random or best_action)
        """
        if random.random() < self.epsilon:
            # Exploration: sample uniformly from action space
            return random.randrange(self.env.num_actions())
        else:
            # Exploitation: use the greedy action
            return best_action


# =============================================================================
# Unit Tests
# =============================================================================

def _run_tests() -> None:
    """Run unit tests for exploration strategies."""
    from utils.test_env import EnvTest
    
    def test_exploration_randomness():
        """Test that exploration produces non-greedy actions."""
        env = EnvTest((5, 5, 1))
        exp_strat = LinearExploration(env, eps_begin=1.0, eps_end=0.0, nsteps=10)
        
        # With epsilon=1.0, should sometimes get action != 0
        found_diff = False
        for _ in range(10):
            action = exp_strat.get_action(best_action=0)
            if action != 0 and action is not None:
                found_diff = True
                break
        
        assert found_diff, "Exploration should produce random actions with epsilon=1.0"
        print("✓ Test exploration randomness: PASSED")

    def test_linear_decay():
        """Test that epsilon decays linearly."""
        env = EnvTest((5, 5, 1))
        exp_strat = LinearExploration(env, eps_begin=1.0, eps_end=0.0, nsteps=10)
        
        exp_strat.update(t=5)
        assert abs(exp_strat.epsilon - 0.5) < 1e-6, f"Expected epsilon=0.5, got {exp_strat.epsilon}"
        print("✓ Test linear decay: PASSED")

    def test_epsilon_floor():
        """Test that epsilon doesn't decay below eps_end."""
        env = EnvTest((5, 5, 1))
        exp_strat = LinearExploration(env, eps_begin=1.0, eps_end=0.5, nsteps=10)
        
        exp_strat.update(t=20)  # Beyond nsteps
        assert exp_strat.epsilon == 0.5, f"Expected epsilon=0.5, got {exp_strat.epsilon}"
        print("✓ Test epsilon floor: PASSED")

    # Run all tests
    test_exploration_randomness()
    test_linear_decay()
    test_epsilon_floor()
    print("\n✓ All exploration strategy tests passed!")


if __name__ == "__main__":
    _run_tests()
