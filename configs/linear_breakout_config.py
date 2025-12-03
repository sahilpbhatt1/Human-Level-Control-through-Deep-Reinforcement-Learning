"""
Configuration for Linear DQN Training

This module defines hyperparameters for training a linear Deep Q-Network
on MinAtar environments. The linear architecture is useful for:
    - Quick prototyping and debugging
    - Baseline comparisons
    - Understanding core DQN components

Hyperparameter Descriptions:
    - Training parameters control the overall learning process
    - Network parameters define model architecture choices
    - Exploration parameters balance exploration vs exploitation
    - Evaluation parameters control how we measure performance

Author: Sahil Bhatt
"""


class config:
    """Linear DQN configuration for MinAtar Breakout."""
    
    # =========================================================================
    # Environment Configuration
    # =========================================================================
    render_train = False          # Render environment during training
    render_test = False           # Render environment during evaluation
    overwrite_render = True       # Overwrite rendering settings
    record = True                 # Record videos during training
    
    # =========================================================================
    # Output Configuration
    # =========================================================================
    output_path = "results/linear_dqn/"
    model_output = output_path + "model.weights"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"
    record_path = output_path + "monitor/"
    
    # =========================================================================
    # Training Configuration
    # =========================================================================
    num_episodes_test = 50        # Episodes per evaluation
    grad_clip = False             # Whether to clip gradients
    saving_freq = 250000          # Steps between model saves
    log_freq = 50                 # Steps between log updates
    eval_freq = 25000             # Steps between evaluations
    record_freq = 250000          # Steps between video recordings
    soft_epsilon = 0.05           # Evaluation exploration rate
    
    # =========================================================================
    # DQN Hyperparameters (from Nature paper)
    # =========================================================================
    
    # Training duration
    nsteps_train = 1000000        # Total environment steps
    
    # Replay buffer
    batch_size = 32               # Mini-batch size for SGD
    buffer_size = 100000          # Replay buffer capacity
    
    # Target network
    target_update_freq = 1000     # Steps between target network updates
    
    # Bellman equation
    gamma = 0.99                  # Discount factor for future rewards
    
    # Learning
    learning_freq = 1             # Steps between gradient updates
    learning_start = 5000         # Steps before learning starts
    
    # Learning rate schedule
    lr_begin = 0.00025            # Initial learning rate
    lr_end = 0.00025              # Final learning rate (no decay)
    lr_nsteps = 500000            # Steps for learning rate decay
    
    # Exploration schedule (epsilon-greedy)
    eps_begin = 1.0               # Initial exploration rate (fully random)
    eps_end = 0.1                 # Final exploration rate
    eps_nsteps = 100000           # Steps for epsilon decay
    
    # State history (for frame stacking, not used in MinAtar)
    state_history = 1
