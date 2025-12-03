"""
Configuration for Nature DQN Training

This module defines hyperparameters for training the convolutional Deep Q-Network
from DeepMind's Nature paper on MinAtar Breakout. These parameters are adapted
from the original paper to work with MinAtar's smaller state space.

Key Hyperparameters (from Mnih et al., 2015):
    - Discount factor (γ): 0.99
    - Learning rate: 0.00025 (RMSProp in original, Adam here)
    - Replay buffer size: 100,000 transitions
    - Target network update: Every 1,000 steps
    - Epsilon schedule: 1.0 → 0.1 over 100,000 steps

Author: Sahil Bhatt
"""


class config:
    """Nature DQN configuration for MinAtar Breakout."""
    
    # =========================================================================
    # Environment Configuration
    # =========================================================================
    env_name = "MinAtar/Breakout-v0"
    render_train = False          # Render during training (slow)
    render_test = False           # Render during evaluation
    overwrite_render = True
    record = True                 # Record evaluation videos
    
    # =========================================================================
    # Output Configuration
    # =========================================================================
    output_path = "results/nature_dqn/"
    model_output = output_path + "model.weights"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"
    record_path = output_path + "monitor/"
    
    # =========================================================================
    # Evaluation and Logging
    # =========================================================================
    num_episodes_test = 50        # Episodes for computing eval score
    grad_clip = False             # Gradient clipping (disabled)
    saving_freq = 250000          # Model checkpoint frequency
    log_freq = 50                 # Progress logging frequency
    eval_freq = 250000            # Evaluation frequency
    record_freq = 250000          # Video recording frequency
    soft_epsilon = 0.05           # Exploration during evaluation
    
    # =========================================================================
    # DQN Core Hyperparameters
    # =========================================================================
    
    # Training duration
    nsteps_train = 1000000        # Total training steps (1M)
    
    # Experience replay
    # The replay buffer stores transitions for training
    # Larger buffers = more diverse samples but more memory
    batch_size = 32               # Mini-batch size
    buffer_size = 100000          # Max transitions stored
    
    # Target network
    # Updating target network less frequently stabilizes training
    # by providing more consistent TD targets
    target_update_freq = 1000     # Steps between target updates
    
    # Bellman equation parameters
    gamma = 0.99                  # Discount factor (γ)
    
    # Learning frequency
    learning_freq = 1             # Gradient updates per env step
    learning_start = 5000         # Fill buffer before learning
    
    # =========================================================================
    # Learning Rate Schedule
    # =========================================================================
    # Original DQN used constant LR with RMSProp
    # We use Adam with optional decay
    lr_begin = 0.00025            # Initial learning rate
    lr_end = 0.00025              # Final learning rate
    lr_nsteps = 500000            # Decay duration
    
    # =========================================================================
    # Exploration Schedule (ε-greedy)
    # =========================================================================
    # Linear annealing from fully random to mostly greedy
    # This ensures sufficient exploration early in training
    eps_begin = 1.0               # Start with random actions
    eps_end = 0.1                 # End with 10% random actions
    eps_nsteps = 100000           # Decay over first 100K steps
    
    # =========================================================================
    # Frame Stacking (not used in MinAtar)
    # =========================================================================
    state_history = 1             # Number of frames to stack
