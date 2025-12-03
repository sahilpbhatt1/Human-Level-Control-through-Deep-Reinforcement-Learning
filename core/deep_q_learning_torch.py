"""
Deep Q-Network (DQN) Implementation in PyTorch

This module provides the core DQN training infrastructure, implementing the
algorithm from DeepMind's Nature paper "Human-level control through deep
reinforcement learning" (Mnih et al., 2015).

The DQN algorithm combines Q-learning with deep neural networks and introduces
two key innovations for stable training:
    1. Experience Replay: Stores transitions in a buffer and samples uniformly
       to break temporal correlations between consecutive training samples
    2. Target Network: Maintains a separate network for computing TD targets,
       updated periodically to provide stable learning signals

Key Components:
    - Q-Network: Neural network approximating Q(s, a) ≈ Q*(s, a)
    - Target Network: Slowly-updated copy for computing TD targets
    - Experience Replay: Buffer storing (s, a, r, s', done) transitions
    - Epsilon-Greedy Exploration: Balances exploration and exploitation

Training Loop:
    1. Select action using ε-greedy policy
    2. Execute action, observe reward and next state
    3. Store transition in replay buffer
    4. Sample mini-batch from buffer
    5. Compute TD targets using target network
    6. Update Q-network via gradient descent on TD error
    7. Periodically update target network

References:
    - Mnih et al. (2015). "Human-level control through deep reinforcement learning"
    - Mnih et al. (2013). "Playing Atari with Deep Reinforcement Learning"

Author: Sahil Bhatt
"""

import torch
import numpy as np
import torch.nn as nn

from typing import Tuple, Optional
from pathlib import Path
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
from core.q_learning import QN


class DQN(QN):
    """
    Deep Q-Network base class for value-based reinforcement learning.
    
    This abstract class provides the training infrastructure for DQN agents.
    Subclasses must implement the following methods:
        - initialize_models(): Define Q-network and target network architectures
        - get_q_values(): Forward pass to compute Q-values
        - update_target(): Copy Q-network weights to target network
        - calc_loss(): Compute TD error loss
        - add_optimizer(): Initialize the optimizer
    
    The class handles:
        - Device management (CPU/GPU)
        - Model building and weight initialization
        - Training step mechanics (sampling, forward pass, backward pass)
        - Target network synchronization
        - TensorBoard logging
        - Model checkpointing
    
    Attributes:
        q_network: Main Q-network for action selection and training
        target_network: Frozen network for computing TD targets
        optimizer: PyTorch optimizer for training
        device: Computation device ('cuda:0' or 'cpu')
        summary_writer: TensorBoard writer for logging metrics
    """
    
    def __init__(self, env, config, logger=None):
        """
        Initialize the DQN agent.
        
        Args:
            env: Environment instance (MinAtar or Gym-like)
            config: Configuration object with hyperparameters
            logger: Optional logger instance
        """
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        
        # Automatically select GPU if available for faster training
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Running model on device {self.device}")
        
        super().__init__(env, config, logger)
        
        # TensorBoard for training visualization
        self.summary_writer = SummaryWriter(self.config.output_path, max_queue=1e5)

    # =========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # =========================================================================

    def initialize_models(self) -> None:
        """
        Define the Q-network and target network architectures.
        
        Must initialize self.q_network and self.target_network as nn.Module
        instances. Both networks should have identical architectures but
        separate parameters.
        
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError

    def get_q_values(self, state: torch.Tensor, network: str) -> torch.Tensor:
        """
        Compute Q-values for all actions given a batch of states.
        
        Performs a forward pass through the specified network.
        
        Args:
            state: Batch of states with shape 
                   (batch_size, img_height, img_width, n_channels * state_history)
            network: Which network to use ('q_network' or 'target_network')
            
        Returns:
            Q-values tensor of shape (batch_size, num_actions)
            
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError

    def update_target(self) -> None:
        """
        Synchronize target network parameters with Q-network.
        
        This "hard update" copies all parameters from q_network to target_network.
        Called periodically (every target_update_freq steps) to provide
        stable TD targets during training.
        
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError

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
        
        Implements: L = E[(r + γ * max_a' Q_target(s', a') - Q(s, a))^2]
        
        Args:
            q_values: Q(s, a) for all actions, shape (batch_size, num_actions)
            target_q_values: Q_target(s', a') for all actions
            actions: Actions taken, shape (batch_size,)
            rewards: Rewards received, shape (batch_size,)
            done_mask: Terminal state indicators (True = terminal)
            
        Returns:
            Scalar loss value
            
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError

    def add_optimizer(self) -> Optimizer:
        """
        Initialize the optimizer for training the Q-network.
        
        Must set self.optimizer to a PyTorch optimizer instance.
        
        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError

    # =========================================================================
    # Model Building and Initialization
    # =========================================================================

    def build(self) -> None:
        """
        Build the DQN model architecture and initialize weights.
        
        This method:
            1. Calls initialize_models() to create network architectures
            2. Loads pre-trained weights if specified in config
            3. Otherwise initializes weights using Xavier initialization
            4. Moves networks to the appropriate device (CPU/GPU)
            5. Initializes the optimizer
        
        Xavier initialization is used because it:
            - Maintains variance across layers
            - Works well with ReLU activations (with gain=sqrt(2))
            - Promotes faster convergence
        """
        # Create network architectures
        self.initialize_models()
        
        # Load pre-trained weights or initialize randomly
        if hasattr(self.config, "load_path"):
            print("Loading parameters from file:", self.config.load_path)
            load_path = Path(self.config.load_path)
            assert load_path.is_file(), f"Provided load_path ({load_path}) does not exist"
            self.q_network.load_state_dict(torch.load(load_path, map_location="cpu"))
            print("Load successful!")
        else:
            print("Initializing parameters randomly")
            
            def init_weights(m):
                """Xavier initialization for linear and conv layers."""
                if hasattr(m, "weight"):
                    # gain=sqrt(2) is optimal for ReLU activations
                    nn.init.xavier_uniform_(m.weight, gain=2 ** (1.0 / 2))
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            
            self.q_network.apply(init_weights)
        
        # Move networks to GPU if available
        self.q_network = self.q_network.to(self.device)
        self.target_network = self.target_network.to(self.device)
        
        # Initialize optimizer
        self.add_optimizer()

    def initialize(self) -> None:
        """
        Final initialization before training.
        
        Synchronizes target network with Q-network to ensure both
        start with identical parameters.
        """
        assert (
            self.q_network is not None and self.target_network is not None
        ), "WARNING: Networks not initialized. Check initialize_models"
        self.update_target()

    # =========================================================================
    # Logging and Checkpointing
    # =========================================================================

    def add_summary(self, latest_loss: float, latest_total_norm: float, t: int) -> None:
        """
        Log training metrics to TensorBoard.
        
        Records loss, gradient norms, reward statistics, and Q-value statistics
        for monitoring training progress and debugging.
        
        Args:
            latest_loss: Most recent training loss
            latest_total_norm: Gradient norm after clipping
            t: Current timestep
        """
        self.summary_writer.add_scalar("loss", latest_loss, t)
        self.summary_writer.add_scalar("grad_norm", latest_total_norm, t)
        self.summary_writer.add_scalar("Avg_Reward", self.avg_reward, t)
        self.summary_writer.add_scalar("Max_Reward", self.max_reward, t)
        self.summary_writer.add_scalar("Std_Reward", self.std_reward, t)
        self.summary_writer.add_scalar("Avg_Q", self.avg_q, t)
        self.summary_writer.add_scalar("Max_Q", self.max_q, t)
        self.summary_writer.add_scalar("Std_Q", self.std_q, t)
        self.summary_writer.add_scalar("Eval_Reward", self.eval_reward, t)

    def save(self) -> None:
        """
        Save model checkpoint.
        
        Saves the Q-network parameters to the path specified in config.
        Target network is not saved as it can be reconstructed by copying Q-network.
        """
        torch.save(self.q_network.state_dict(), self.config.model_output)

    # =========================================================================
    # Action Selection
    # =========================================================================

    def get_best_action(self, state: torch.Tensor) -> Tuple[int, np.ndarray]:
        """
        Select the greedy action based on current Q-values.
        
        Computes Q(s, a) for all actions and returns the one with highest value.
        This is used during evaluation and as the "best action" input to
        the exploration strategy during training.
        
        Args:
            state: Current observation (can be single state or batch)
            
        Returns:
            action: Index of the action with highest Q-value
            action_values: Q-values for all actions (for logging/debugging)
        """
        with torch.no_grad():
            s = torch.Tensor(state).float()
            action_values = (
                self.get_q_values(s, "q_network").squeeze().to("cpu").tolist()
            )
        action = np.argmax(action_values)
        return action, action_values

    # =========================================================================
    # Training Step
    # =========================================================================

    def update_step(self, t: int, replay_buffer, lr: float) -> Tuple[float, float]:
        """
        Perform a single training step of the DQN algorithm.
        
        This implements the core DQN training:
            1. Sample mini-batch from replay buffer
            2. Compute Q-values for current states (Q-network)
            3. Compute target Q-values for next states (target network)
            4. Calculate TD loss: L = (r + γ * max Q_target - Q)²
            5. Backpropagate and update Q-network parameters
            6. Optionally clip gradients for stability
        
        Args:
            t: Current timestep (for logging)
            replay_buffer: ReplayBuffer instance to sample from
            lr: Current learning rate
            
        Returns:
            loss: TD error loss value
            total_norm: Gradient norm (for monitoring)
        """
        # Sample a mini-batch of transitions
        self.timer.start("update_step/replay_buffer.sample")
        s_batch, sp_batch, a_batch, r_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size
        )
        self.timer.end("update_step/replay_buffer.sample")

        # Validate network initialization
        assert (
            self.q_network is not None and self.target_network is not None
        ), "WARNING: Networks not initialized. Check initialize_models"
        assert (
            self.optimizer is not None
        ), "WARNING: Optimizer not initialized. Check add_optimizer"

        # Convert done mask to boolean
        self.timer.start("update_step/converting_tensors")
        done_mask_batch = done_mask_batch.bool()
        self.timer.end("update_step/converting_tensors")

        # Reset gradients
        self.timer.start("update_step/zero_grad")
        self.optimizer.zero_grad()
        self.timer.end("update_step/zero_grad")

        # Forward pass through Q-network for current states
        self.timer.start("update_step/forward_pass_q")
        q_values = self.get_q_values(s_batch, "q_network")
        self.timer.end("update_step/forward_pass_q")

        # Forward pass through target network for next states (no gradients)
        self.timer.start("update_step/forward_pass_target")
        with torch.no_grad():
            target_q_values = self.get_q_values(sp_batch, "target_network")
        self.timer.end("update_step/forward_pass_target")

        # Compute TD loss
        self.timer.start("update_step/loss_calc")
        loss = self.calc_loss(
            q_values, target_q_values, a_batch, r_batch, done_mask_batch
        )
        self.timer.end("update_step/loss_calc")
        
        # Backward pass
        self.timer.start("update_step/loss_backward")
        loss.backward()
        self.timer.end("update_step/loss_backward")

        # Gradient clipping for training stability
        if self.config.grad_clip:
            self.timer.start("update_step/grad_clip")
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), self.config.clip_val
            ).item()
            self.timer.end("update_step/grad_clip")
        else:
            total_norm = 0

        # Update parameters
        self.timer.start("update_step/optimizer")
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.timer.end("update_step/optimizer")
        
        return loss.item(), total_norm

    def update_target_params(self) -> None:
        """
        Update target network parameters.
        
        Wrapper method that calls update_target() to synchronize
        target network with Q-network.
        """
        self.update_target()
