#!/usr/bin/env python3
"""
Training Script for Nature DQN on MinAtar Breakout

This script trains the convolutional Deep Q-Network architecture from DeepMind's
Nature paper "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
on the MinAtar Breakout environment.

The Nature DQN architecture uses convolutional layers to process visual inputs,
enabling the agent to learn directly from pixel observations without manual
feature engineering. This is a key capability for applying RL to real-world
problems where state representations are not hand-crafted.

Training Details:
    - Environment: MinAtar Breakout (10x10 grid, 4 channels)
    - Architecture: Conv2D(16, 3x3) -> ReLU -> FC(128) -> ReLU -> FC(actions)
    - Algorithm: Deep Q-Learning with experience replay and target network
    - Exploration: Epsilon-greedy with linear annealing

The training runs multiple independent trials to compute statistically reliable
performance estimates with confidence intervals.

Usage:
    python train_nature.py

Output:
    - Model weights saved to results/nature_dqn/
    - TensorBoard logs for monitoring training progress
    - Score plots showing learning curves with error bands

Author: Sahil Bhatt
"""

import logging
from pathlib import Path

from minatar import Environment

from models.nature_dqn import NatureDQN
from utils.exploration import LinearExploration, LinearSchedule
from utils.general import export_mean_plot
from configs.nature_breakout_config import config


def train_nature_dqn(num_runs: int = 3) -> None:
    """
    Train a Nature DQN agent on MinAtar Breakout.
    
    Runs multiple independent training trials to provide statistically
    meaningful performance estimates. Results are aggregated to show
    mean performance with standard deviation bands.
    
    Args:
        num_runs: Number of independent training runs (default: 3)
    """
    print("=" * 60)
    print("Training Nature DQN on MinAtar Breakout")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Training steps: {config.nsteps_train:,}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Replay buffer size: {config.buffer_size:,}")
    print(f"  - Learning rate: {config.lr_begin}")
    print(f"  - Epsilon: {config.eps_begin} -> {config.eps_end} over {config.eps_nsteps:,} steps")
    print(f"  - Discount factor (γ): {config.gamma}")
    print(f"  - Target network update: every {config.target_update_freq:,} steps")
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
        
        # Initialize exploration schedule
        # Epsilon decays from 1.0 (fully random) to 0.1 (mostly greedy)
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
        model.run(exp_schedule, lr_schedule, run_idx=run_idx)
    
    # Generate aggregate plot showing mean ± std across runs
    print("\nGenerating aggregate performance plot...")
    export_mean_plot("Scores", config.plot_output, config.output_path)
    print(f"Results saved to {config.output_path}")


def main():
    """Entry point for Nature DQN training."""
    # Suppress matplotlib font warnings
    logging.getLogger("matplotlib.font_manager").disabled = True
    
    train_nature_dqn(num_runs=3)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Monitor training: tensorboard --logdir=results/")
    print("  2. View learning curves: results/nature_dqn/scores.png")
    print("  3. Trained model weights: results/nature_dqn/model.weights")


if __name__ == "__main__":
    main()
