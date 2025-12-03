"""
Image Preprocessing for Atari Environments

This module provides preprocessing functions to convert raw Atari frames
into a format suitable for neural network input. These transformations
follow the approach from the DQN paper (Mnih et al., 2015).

Preprocessing Steps:
    1. Convert RGB to grayscale (reduces input dimensionality)
    2. Crop irrelevant portions of the frame
    3. Downsample to reduce computational requirements
    4. Normalize pixel values

The preprocessing reduces the observation from (210, 160, 3) to (80, 80, 1),
significantly reducing memory usage and enabling faster training.

Note: These functions are for standard Atari environments. MinAtar uses
smaller native observations that don't require this preprocessing.

References:
    - Mnih et al. (2015). "Human-level control through deep reinforcement learning"
    - Karpathy's Pong preprocessing: http://karpathy.github.io/2016/05/31/rl/

Author: Sahil Bhatt
"""

import numpy as np
from typing import Tuple


def greyscale(state: np.ndarray) -> np.ndarray:
    """
    Convert Atari frame to grayscale and downsample.
    
    Preprocessing pipeline:
        1. Convert RGB to grayscale using standard weights
        2. Crop to remove score/lives display (rows 35-195)
        3. Downsample by factor of 2 (160x160 -> 80x80)
    
    Args:
        state: Raw Atari frame of shape (210, 160, 3) as uint8
        
    Returns:
        Preprocessed frame of shape (80, 80, 1) as uint8
    """
    state = np.reshape(state, [210, 160, 3]).astype(np.float32)

    # Standard luminance weights for grayscale conversion
    state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114

    # Crop and downsample (Karpathy preprocessing)
    state = state[35:195]       # Crop top and bottom
    state = state[::2, ::2]     # Downsample by 2

    state = state[:, :, np.newaxis]
    return state.astype(np.uint8)


def blackandwhite(state: np.ndarray) -> np.ndarray:
    """
    Convert Atari frame to binary (black and white).
    
    More aggressive preprocessing that binarizes the frame,
    which can be useful for games with simple graphics.
    
    Args:
        state: Raw Atari frame of shape (210, 160, 3)
        
    Returns:
        Binary frame of shape (80, 80, 1) with values 0 or 1
    """
    # Erase common background colors
    state[state == 144] = 0
    state[state == 109] = 0
    state[state != 0] = 1

    # Crop and downsample
    state = state[35:195]
    state = state[::2, ::2, 0]

    state = state[:, :, np.newaxis]
    return state.astype(np.uint8)