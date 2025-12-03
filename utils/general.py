"""
General Utilities for Deep Reinforcement Learning

This module provides utility functions for:
    - Logging: File and console logging setup
    - Plotting: Training curve visualization with mean and std
    - Progress tracking: Progress bar for training loops

These utilities support experiment tracking and reproducibility,
which are essential for rigorous machine learning research.

Author: Sahil Bhatt
"""

import time
import sys
import logging
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("agg")  # Use non-interactive backend for servers
import matplotlib.pyplot as plt
import pickle


def export_mean_plot(ylabel: str, filename: str, output_folder: str) -> None:
    """
    Generate a plot showing mean performance across multiple runs with std bands.
    
    Loads score files from multiple training runs and creates a publication-quality
    plot showing mean performance with standard deviation shading.
    
    Args:
        ylabel: Label for the y-axis (e.g., "Scores")
        filename: Output filename for the plot
        output_folder: Directory containing the score pickle files
    """
    # Load scores from all runs
    with open(output_folder + "scores_1.pkl", "rb") as f:
        scores_1 = np.array(pickle.load(f))
    with open(output_folder + "scores_2.pkl", "rb") as f:
        scores_2 = np.array(pickle.load(f))
    with open(output_folder + "scores_3.pkl", "rb") as f:
        scores_3 = np.array(pickle.load(f))

    # Compute mean and standard deviation
    ys = (scores_1 + scores_2 + scores_3) / 3
    ys = list(ys)
    std = np.std([scores_1, scores_2, scores_3], axis=0)
    
    # Create plot with confidence bands
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(ys)), ys, linewidth=2)
    plt.fill_between(range(len(ys)), ys - std, ys + std, alpha=0.2)
    plt.xlabel("Evaluation Episode", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title("Training Performance (Mean ± Std)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def export_plot(ys: List[float], ylabel: str, filename: str) -> None:
    """
    Export a simple line plot of training scores.
    
    Args:
        ys: List of y-values to plot
        ylabel: Label for the y-axis
        filename: Output filename for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(ys)), ys, linewidth=2)
    plt.xlabel("Evaluation Episode", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title("Training Progress", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def get_logger(filename: str) -> logging.Logger:
    """
    Create and configure a logger for training experiments.
    
    Sets up logging to both file and console for experiment tracking.
    
    Args:
        filename: Path to the log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return logger


class Progbar:
    """
    Progress bar for tracking training iterations.
    
    Displays a visual progress bar with metrics including loss, reward,
    and other training statistics. Adapted from Keras progress bar.
    
    Features:
        - Real-time progress visualization
        - Support for multiple metrics with moving averages
        - ETA estimation
        - Customizable display width
    
    Example:
        >>> prog = Progbar(target=100000)
        >>> prog.update(t, exact=[("Loss", 0.5), ("Reward", 10.2)])
    
    Attributes:
        target: Total number of steps
        width: Width of the progress bar in characters
        seen_so_far: Current step count
    """

    def __init__(
        self, 
        target: int, 
        width: int = 30, 
        verbose: int = 1, 
        discount: float = 0.9
    ) -> None:
        """
        Initialize the progress bar.
        
        Args:
            target: Total number of steps expected
            width: Visual width of the progress bar
            verbose: Verbosity level (0: silent, 1: progress bar, 2: summary)
            discount: Exponential moving average discount factor
        """
        self.width = width
        self.target = target
        self.sum_values = {}
        self.exp_avg = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose
        self.discount = discount

    def reset_start(self) -> None:
        """Reset the start time for ETA calculation."""
        self.start = time.time()

    def update(
        self, 
        current: int, 
        values: List = [], 
        exact: List = [], 
        strict: List = [], 
        exp_avg: List = [], 
        base: int = 0
    ) -> None:
        """
        Update the progress bar with new metrics.
        
        Args:
            current: Current step index
            values: List of (name, value) tuples for running averages
            exact: List of (name, value) tuples for exact values
            strict: List of (name, value) tuples for strict values
            exp_avg: List of (name, value) tuples for exponential averages
            base: Base value for progress calculation
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [
                    v * (current - self.seen_so_far),
                    current - self.seen_so_far,
                ]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += current - self.seen_so_far
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        for k, v in strict:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = v
        for k, v in exp_avg:
            if k not in self.exp_avg:
                self.exp_avg[k] = v
            else:
                self.exp_avg[k] *= self.discount
                self.exp_avg[k] += (1 - self.discount) * v

        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = "%%%dd/%%%dd [" % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += "=" * (prog_width - 1)
                if current < self.target:
                    bar += ">"
                else:
                    bar += "="
            bar += "." * (self.width - prog_width)
            bar += "]"
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / (current - base)
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ""
            if current < self.target:
                info += " - ETA: %ds" % eta
            else:
                info += " - %ds" % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += " - %s: %.4f" % (
                        k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]),
                    )
                else:
                    info += " - %s: %s" % (k, self.sum_values[k])

            for k, v in self.exp_avg.items():
                info += " - %s: %.4f" % (k, v)

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += (prev_total_width - self.total_width) * " "

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = "%ds" % (now - self.start)
                for k in self.unique_values:
                    info += " - %s: %.4f" % (
                        k,
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]),
                    )
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)
