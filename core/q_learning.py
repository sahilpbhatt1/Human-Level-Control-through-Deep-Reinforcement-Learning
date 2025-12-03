"""
Q-Learning Base Class and Training Infrastructure

This module provides the foundational Q-learning agent class and training loop
for reinforcement learning experiments. It implements the core training loop
used by both tabular and deep Q-learning variants.

The training process follows the standard RL interaction loop:
    1. Agent observes state s from environment
    2. Agent selects action a using policy (e.g., ε-greedy)
    3. Environment transitions to state s' and returns reward r
    4. Agent stores transition (s, a, r, s', done) in replay buffer
    5. Agent updates Q-function using sampled experiences
    6. Repeat until training completes

Key Components:
    - Timer: Performance profiling for optimization
    - QN: Base Q-learning agent with training loop
    - Experience collection and replay buffer management
    - Evaluation and logging infrastructure

Design Principles:
    - Modular design separating environment interaction from learning
    - Configurable hyperparameters via config objects
    - Built-in support for TensorBoard logging
    - Checkpoint saving and loading

Author: Sahil Bhatt
"""

import os
import gym
from minatar import Environment
import numpy as np
import time
import sys
import torch
from collections import deque, defaultdict
import random
import pickle
from typing import Optional, List, Tuple

from utils.general import get_logger, Progbar, export_plot
from utils.replay_buffer import ReplayBuffer
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv


class Timer:
    """
    Performance profiling utility for measuring execution time.
    
    Tracks cumulative time spent in different code sections to identify
    bottlenecks and optimize training performance. Useful for profiling
    replay buffer operations, neural network forward/backward passes,
    and environment interactions.
    
    Usage:
        timer = Timer(enabled=True)
        timer.start("forward_pass")
        # ... do forward pass ...
        timer.end("forward_pass")
        timer.print_stat()  # Print timing summary
    
    Attributes:
        enabled: Whether timing is active (disabled for production)
        category_sec_avg: Dict mapping category names to timing stats
    """
    
    def __init__(self, enabled: bool = False) -> None:
        """
        Initialize the timer.
        
        Args:
            enabled: If True, collect timing statistics. 
                     If False, timing calls are no-ops for zero overhead.
        """
        self.enabled = enabled
        # Format: {category: [total_secs, latest_start, num_calls]}
        self.category_sec_avg = defaultdict(lambda: [0.0, 0.0, 0])

    def start(self, category: str) -> None:
        """Start timing a code section."""
        if self.enabled:
            stat = self.category_sec_avg[category]
            stat[1] = time.perf_counter()
            stat[2] += 1

    def end(self, category: str) -> None:
        """End timing a code section and accumulate time."""
        if self.enabled:
            stat = self.category_sec_avg[category]
            stat[0] += time.perf_counter() - stat[1]

    def print_stat(self) -> None:
        """Print timing statistics for all tracked categories."""
        if self.enabled:
            print("Printing timer stats:")
            for key, val in self.category_sec_avg.items():
                if val[2] > 0:
                    avg_time = val[0] / val[2]
                    print(f"  {key}: total={val[0]:.3f}s, calls={val[2]}, avg={avg_time:.6f}s")

    def reset_stat(self) -> None:
        """Reset all timing statistics."""
        if self.enabled:
            print("Resetting timer stats")
            for val in self.category_sec_avg.values():
                val[0], val[1], val[2] = 0.0, 0.0, 0


class QN:
    """
    Abstract base class for Q-Learning agents.
    
    Provides the training infrastructure shared by all Q-learning variants:
        - Configuration management
        - Training loop with environment interaction
        - Experience replay integration
        - Evaluation and logging
        - Model checkpointing
    
    Subclasses must implement:
        - get_best_action(): Compute greedy action from Q-values
        - update_step(): Perform one gradient update
        - update_target_params(): Sync target network (if applicable)
    
    The training loop collects experiences, manages the replay buffer,
    and orchestrates the learning process while tracking statistics
    for monitoring convergence.
    
    Attributes:
        env: The training environment
        config: Hyperparameter configuration
        logger: Logging instance
        timer: Performance profiler
    """

    def __init__(self, env, config, logger=None):
        """
        Initialize the Q-learning agent.
        
        Sets up output directories, logging, and builds the model.
        
        Args:
            env: Environment instance (MinAtar or Gym-compatible)
            config: Configuration object with hyperparameters
            logger: Optional custom logger (creates default if None)
        """
        # Create output directory for results
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        self.config = config
        self.logger = logger if logger is not None else get_logger(config.log_path)
        self.env = env
        
        # Timer for profiling (disabled by default for performance)
        self.timer = Timer(enabled=False)

        # Build the model (networks, optimizer, etc.)
        self.build()

    def build(self) -> None:
        """Build the model. Override in subclasses."""
        pass

    @property
    def policy(self):
        """Return the agent's policy function: state -> action."""
        return lambda state: self.get_action(state)

    def save(self) -> None:
        """Save model parameters. Override in subclasses."""
        pass

    def initialize(self) -> None:
        """Initialize variables if necessary. Override in subclasses."""
        pass

    def get_best_action(self, state) -> Tuple[int, List[float]]:
        """
        Return the greedy action according to the Q-network.
        
        Args:
            state: Current observation
            
        Returns:
            action: Greedy action index
            q_values: Q-values for all actions
        """
        raise NotImplementedError

    def get_action(self, state) -> int:
        """
        Select an action using soft epsilon-greedy during evaluation.
        
        Uses a small exploration rate (soft_epsilon) during evaluation
        to provide some stochasticity for more robust performance estimates.
        
        Args:
            state: Current observation
            
        Returns:
            Selected action index
        """
        if np.random.random() < self.config.soft_epsilon:
            return torch.tensor([[random.randrange(self.env.num_actions())]])
        else:
            return self.get_best_action(state)[0]

    def update_target_params(self) -> None:
        """Update target network parameters. Override in subclasses."""
        pass

    # =========================================================================
    # Training Statistics
    # =========================================================================

    def init_averages(self) -> None:
        """Initialize statistics for tracking training progress."""
        self.avg_reward = -21.0
        self.max_reward = -21.0
        self.std_reward = 0

        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0

        self.eval_reward = -21.0

    def update_averages(
        self, 
        rewards: deque, 
        max_q_values: deque, 
        q_values: deque, 
        scores_eval: List[float]
    ) -> None:
        """
        Update running statistics for logging.
        
        Args:
            rewards: Recent episode rewards
            max_q_values: Recent maximum Q-values
            q_values: Recent Q-values
            scores_eval: Evaluation scores
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_q = np.mean(max_q_values)
        self.avg_q = np.mean(q_values)
        self.std_q = np.sqrt(np.var(q_values) / len(q_values))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def add_summary(self, latest_loss: float, latest_total_norm: float, t: int) -> None:
        """Log training metrics. Override in subclasses for TensorBoard."""
        pass

    # =========================================================================
    # Main Training Loop
    # =========================================================================

    # =========================================================================
    # Main Training Loop
    # =========================================================================

    def train(self, exp_schedule, lr_schedule, run_idx: int) -> None:
        """
        Main training loop for Q-learning.
        
        Implements the standard deep RL training process:
            1. Initialize replay buffer and statistics
            2. For each timestep:
                a. Select action using exploration strategy
                b. Execute action in environment
                c. Store transition in replay buffer
                d. Perform gradient update (after warmup)
                e. Update target network periodically
                f. Log statistics and save checkpoints
        
        Args:
            exp_schedule: Exploration schedule (e.g., LinearExploration)
            lr_schedule: Learning rate schedule
            run_idx: Index for multiple runs (for result aggregation)
        """
        # Initialize replay buffer and tracking variables
        replay_buffer = ReplayBuffer(self.config.buffer_size)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        self.init_averages()

        t = last_eval = last_record = 0
        scores_eval = []
        scores_eval += [self.evaluate()]

        prog = Progbar(target=self.config.nsteps_train)

        # Main training loop
        while t < self.config.nsteps_train:
            total_reward = 0
            self.timer.start("env.reset")
            # state, _ = self.env.reset()
            state = self.env.reset()
            state = self.env.state()
            # state = torch.Tensor(state).permute(2, 0, 1).unsqueeze(0).float()
            state = torch.Tensor(state).unsqueeze(0).float()
            self.timer.end("env.reset")
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                if self.config.render_train:
                    self.env.render()

                # chose action according to current Q and exploration
                self.timer.start("get_action")
                best_action, q_vals = self.get_best_action(state)
                action = exp_schedule.get_action(best_action)
                self.timer.end("get_action")

                # store q values
                max_q_values.append(max(q_vals))
                q_values += list(q_vals)

                # perform action in env
                self.timer.start("env.step")
                # new_state, reward, done, _, _ = self.env.step(action)
                reward, done = self.env.act(action)
                new_state = self.env.state()
                self.timer.end("env.step")

                # store the transition
                self.timer.start("replay_buffer.store_effect")
                new_state = (
                    # torch.Tensor(new_state).permute(2, 0, 1).unsqueeze(0).float()
                    torch.Tensor(new_state)
                    .unsqueeze(0)
                    .float()
                )
                replay_buffer.add(
                    state,
                    new_state,
                    torch.Tensor([action]).float(),
                    torch.Tensor([[reward]]).float(),
                    torch.Tensor([[done]]).float(),
                )
                state = new_state
                self.timer.end("replay_buffer.store_effect")

                # perform a training step
                self.timer.start("train_step")
                loss_eval, grad_eval = self.train_step(
                    t, replay_buffer, lr_schedule.epsilon
                )
                self.timer.end("train_step")

                # logging stuff
                if (
                    (t > self.config.learning_start)
                    and (t % self.config.log_freq == 0)
                    and (t % self.config.learning_freq == 0)
                ):
                    self.timer.start("logging")
                    self.update_averages(rewards, max_q_values, q_values, scores_eval)
                    self.add_summary(loss_eval, grad_eval, t)
                    exp_schedule.update(t)
                    lr_schedule.update(t)
                    if len(rewards) > 0:
                        prog.update(
                            t + 1,
                            exact=[
                                ("Loss", loss_eval),
                                ("Avg_R", self.avg_reward),
                                ("Max_R", np.max(rewards)),
                                ("eps", exp_schedule.epsilon),
                                ("Grads", grad_eval),
                                ("Max_Q", self.max_q),
                                ("lr", lr_schedule.epsilon),
                            ],
                            base=self.config.learning_start,
                        )
                    self.timer.end("logging")
                elif (t < self.config.learning_start) and (
                    t % self.config.log_freq == 0
                ):
                    sys.stdout.write(
                        "\rPopulating the memory {}/{}...".format(
                            t, self.config.learning_start
                        )
                    )
                    sys.stdout.flush()
                    prog.reset_start()

                # count reward
                total_reward += reward
                if done or t >= self.config.nsteps_train:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                self.timer.start("eval")
                scores_eval += [self.evaluate()]
                self.timer.end("eval")
                self.timer.print_stat()
                self.timer.reset_stat()

            if (
                (t > self.config.learning_start)
                and self.config.record
                and (last_record > self.config.record_freq)
            ):
                self.logger.info("Recording...")
                last_record = 0
                self.timer.start("recording")
                self.record()
                self.timer.end("recording")

        # last words
        self.logger.info("- Training done.")
        self.save()
        scores_eval += [self.evaluate()]
        with open(self.config.output_path + "scores_{}.pkl".format(run_idx), "wb") as f:
            pickle.dump(scores_eval, f)
        export_plot(scores_eval, "Scores", self.config.plot_output)

    def train_step(self, t, replay_buffer, lr):
        """
        Perform training step

        Args:
            t: (int) nths step
            replay_buffer: buffer for sampling
            lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        if t > self.config.learning_start and t % self.config.learning_freq == 0:
            self.timer.start("train_step/update_step")
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)
            self.timer.end("train_step/update_step")

        # occasionaly update target network with q network
        if t % self.config.target_update_freq == 0:
            self.timer.start("train_step/update_param")
            self.update_target_params()
            self.timer.end("train_step/update_param")

        # occasionaly save the weights
        if t % self.config.saving_freq == 0:
            self.timer.start("train_step/save")
            self.save()
            self.timer.end("train_step/save")

        return loss_eval, grad_eval

    def evaluate(self, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        # replay memory to play
        replay_buffer = ReplayBuffer(self.config.buffer_size)
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            # state, _ = env.reset()
            state = env.reset()
            state = env.state()
            while True:
                if self.config.render_test:
                    env.render()

                action = self.get_action(state[None])

                # perform action in env
                # new_state, reward, done, _, _ = env.step(action)
                reward, done = env.act(action)
                new_state = env.state()

                # store in replay memory
                replay_buffer.add(state, new_state, action, reward, done)
                state = new_state

                # count reward
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
                avg_reward, sigma_reward
            )
            self.logger.info(msg)

        return avg_reward

    def record(self):
        """
        Re create an env and record a video for one episode
        """
        # env = gym.make(self.config.env_name)
        env = Environment("breakout")
        # env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True,
        # resume=True)
        # env = MaxAndSkipEnv(env, skip=self.config.skip_frame)
        # env = PreproWrapper(
        #     env,
        #     prepro=greyscale,
        #     shape=(80, 80, 1),
        #     overwrite_render=self.config.overwrite_render,
        # )
        self.evaluate(env, 1)

    def run(self, exp_schedule, lr_schedule, run_idx):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()

        # record one game at the beginning
        if self.config.record:
            self.record()

        # model
        self.train(exp_schedule, lr_schedule, run_idx)

        # record one game at the end
        if self.config.record:
            self.record()
