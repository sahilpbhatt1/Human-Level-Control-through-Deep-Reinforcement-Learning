"""
Environment Wrappers for Atari Games

This module provides Gym wrappers that modify environment behavior for
deep reinforcement learning. These wrappers implement standard preprocessing
techniques used in the DQN literature.

Wrappers Included:
    - MaxAndSkipEnv: Frame skipping with max pooling for temporal abstraction
    - PreproWrapper: Image preprocessing (grayscale, crop, resize)

Frame skipping reduces the effective episode length and provides
temporally extended actions, which are important for Atari games
where single frames may not capture important motion information.

Note: These wrappers are designed for OpenAI Gym Atari environments.
MinAtar environments have a different interface and don't require these wrappers.

References:
    - Mnih et al. (2015). "Human-level control through deep reinforcement learning"

Author: Sahil Bhatt
"""

import numpy as np
import gym
from gym import spaces
from collections import deque
from typing import Callable, Tuple, Optional

from utils.viewer import SimpleImageViewer


class MaxAndSkipEnv(gym.Wrapper):
    """
    Frame skipping wrapper with max pooling.
    
    Repeats each action for `skip` frames and returns the maximum pixel
    value over the last 2 frames. Max pooling addresses flickering in
    Atari games where sprites may be invisible every other frame.
    
    This provides:
        1. Reduced computational cost (fewer decisions per episode)
        2. Temporally extended actions
        3. Handles Atari flickering artifacts
    
    Attributes:
        _skip: Number of frames to skip
        _obs_buffer: Buffer storing last 2 observations for max pooling
    """
    
    def __init__(self, env: gym.Env = None, skip: int = 4) -> None:
        """
        Initialize the wrapper.
        
        Args:
            env: Base Gym environment to wrap
            skip: Number of frames to repeat each action (default: 4)
        """
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action for skip frames with max pooling.
        
        Args:
            action: Action to execute
            
        Returns:
            observation: Max over last 2 frames
            total_reward: Sum of rewards over skipped frames
            done: Whether episode terminated
            info: Additional info from environment
        """
        total_reward = 0.0
        done = None
        
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        # Max over last 2 frames to handle flickering
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self) -> np.ndarray:
        """Reset environment and clear observation buffer."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class PreproWrapper(gym.Wrapper):
    """
    Preprocessing wrapper for Atari environments.
    
    Applies a preprocessing function (e.g., grayscale conversion) to
    observations and optionally overrides rendering to show preprocessed frames.
    
    Attributes:
        prepro: Preprocessing function to apply to observations
        obs: Most recent preprocessed observation
        overwrite_render: Whether to render preprocessed frames
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        prepro: Callable, 
        shape: Tuple[int, ...], 
        overwrite_render: bool = True, 
        high: int = 255
    ) -> None:
        """
        Initialize the preprocessing wrapper.
        
        Args:
            env: Base Gym environment
            prepro: Preprocessing function (e.g., greyscale)
            shape: Shape of preprocessed observation
            overwrite_render: If True, render preprocessed frames
            high: Maximum pixel value after preprocessing
        """
        super(PreproWrapper, self).__init__(env)
        self.overwrite_render = overwrite_render
        self.viewer = None
        self.prepro = prepro
        self.observation_space = spaces.Box(
            low=0, high=high, shape=shape, dtype=np.uint8
        )
        self.high = high
        self.obs = None

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and preprocess observation."""
        obs, reward, done, info = self.env.step(action)
        self.obs = self.prepro(obs)
        return self.obs, reward, done, info

    def reset(self) -> np.ndarray:
        """Reset environment and preprocess initial observation."""
        self.obs = self.prepro(self.env.reset())
        return self.obs

    def _render(self, mode: str = 'human', close: bool = False):
        """
        Render the preprocessed observation.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            close: Whether to close the viewer
        """
        if self.overwrite_render:
            if close:
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                return
                
            img = self.obs
            if mode == 'rgb_array':
                return img
            elif mode == 'human':
                if self.viewer is None:
                    self.viewer = SimpleImageViewer()
                self.viewer.imshow(img)
        else:
            super(PreproWrapper, self)._render(mode, close)
