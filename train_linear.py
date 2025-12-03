#!/usr/bin/env python3
"""
Training Script for Linear DQN on MinAtar Breakout

This script trains a linear Deep Q-Network on the MinAtar Breakout environment.
The linear architecture serves as a baseline for comparing against more complex
convolutional architectures.

MinAtar Breakout is a simplified version of Atari Breakout with:
    - 10x10 pixel grid (vs 210x160 in original Atari)
    - 4 channels representing different game objects
    - 6 possible actions

The training runs multiple independent trials to compute mean performance
with confidence intervals, following best practices for RL evaluation.

Usage:
    python train_linear.py

Output:
    - Model weights saved to results/linear_dqn/
    - Training logs for TensorBoard visualization
    - Score plots showing learning progress

Author: Sahil Bhatt
"""

import logging
from pathlib import Path

from minatar import Environment

from models.linear_dqn import LinearDQN
from utils.exploration import LinearExploration, LinearSchedule
from utils.general import export_mean_plot
from configs.linear_breakout_config import config


def train_linear_dqn(num_runs: int = 3) -> None:
    """
    Train a Linear DQN agent on MinAtar Breakout.
    
    Runs multiple independent training trials and aggregates results
    to provide statistically meaningful performance estimates.
    
    Args:
        num_runs: Number of independent training runs (default: 3)
    """
    print("=" * 60)
    print("Training Linear DQN on MinAtar Breakout")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Training steps: {config.nsteps_train:,}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Buffer size: {config.buffer_size:,}")
    print(f"  - Learning rate: {config.lr_begin} -> {config.lr_end}")
    print(f"  - Epsilon: {config.eps_begin} -> {config.eps_end}")
    print(f"  - Discount factor (γ): {config.gamma}")
    print(f"  - Target update frequency: {config.target_update_freq}")
    print(f"  - Number of runs: {num_runs}")
    print()
    
    # Create output directory
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment
    env = Environment("breakout")
    
    for run_idx in range(1, num_runs + 1):
        print(f"\n{'='*60}")
        print(f"Starting Run {run_idx}/{num_runs}")
        print(f"{'='*60}\n")
        
        # Initialize exploration schedule (epsilon-greedy with linear decay)
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
        model.run(exp_schedule, lr_schedule, run_idx=run_idx)
    
    # Generate aggregate plot with mean and standard deviation
    print("\nGenerating aggregate performance plot...")
    export_mean_plot("Scores", config.plot_output, config.output_path)
    print(f"Results saved to {config.output_path}")


def main():
    """Entry point for linear DQN training."""
    # Suppress matplotlib font warnings
    logging.getLogger("matplotlib.font_manager").disabled = True
    
    train_linear_dqn(num_runs=3)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. View training logs: tensorboard --logdir=results/")
    print("  2. Check score plots in results/linear_dqn/scores.png")
    print("  3. Try Nature DQN for better performance: python train_nature.py")


if __name__ == "__main__":
    main()
