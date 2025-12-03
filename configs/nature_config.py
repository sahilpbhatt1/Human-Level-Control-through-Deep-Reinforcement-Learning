"""
Configuration for Nature DQN on Test Environment

Fast configuration for unit testing and debugging the convolutional DQN.
Uses reduced training steps and buffer size for rapid iteration.

Author: Sahil Bhatt
"""


class config:
    """Test configuration for Nature DQN with small environment."""
    
    # Environment
    render_train = False
    render_test = False
    overwrite_render = True
    record = False
    high = 255.0
    
    # Output
    output_path = "results/nature_test/"
    model_output = output_path + "model.weights"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"
    
    # Training
    num_episodes_test = 20
    grad_clip = True
    clip_val = 10
    saving_freq = 5000
    log_freq = 50
    eval_freq = 1000
    soft_epsilon = 0
    
    # Hyperparameters (reduced for fast testing)
    nsteps_train = 100000
    batch_size = 32
    buffer_size = 500
    target_update_freq = 500
    gamma = 0.99
    learning_freq = 4
    state_history = 1
    lr_begin = 0.00025
    lr_end = 0.0001
    lr_nsteps = nsteps_train / 2
    eps_begin = 1
    eps_end = 0.01
    eps_nsteps = nsteps_train / 2
    learning_start = 200
