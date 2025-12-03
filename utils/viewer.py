"""
Simple Image Viewer for Rendering

This module provides a lightweight image viewer for visualizing
environment observations and agent behavior during training and evaluation.

Based on OpenAI Gym's rendering utilities but modified to support
different image formats (RGB and grayscale).

Author: Sahil Bhatt
"""

import pyglet
from typing import Optional


class SimpleImageViewer:
    """
    Simple image viewer window using pyglet.
    
    Supports both RGB and grayscale images for visualization.
    Useful for debugging and monitoring agent behavior.
    
    Attributes:
        window: Pyglet window instance
        isopen: Whether the window is currently open
        width: Image width
        height: Image height
    """
    
    def __init__(self, display: Optional = None) -> None:
        """
        Initialize the viewer.
        
        Args:
            display: Display to use (for remote rendering)
        """
        self.window = None
        self.isopen = False
        self.display = display
        self.width = 0
        self.height = 0

    def imshow(self, arr) -> None:
        """
        Display an image in the viewer window.
        
        Creates the window on first call, then updates the displayed image.
        Supports both RGB (3 channels) and grayscale (1 channel) images.
        
        Args:
            arr: Image array of shape (height, width, channels)
        """
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(
                width=width, 
                height=height, 
                display=self.display
            )
            self.width = width
            self.height = height
            self.isopen = True

        # Determine image format based on channels
        nchannels = arr.shape[-1]
        if nchannels == 1:
            _format = "I"  # Grayscale
        elif nchannels == 3:
            _format = "RGB"
        else:
            raise NotImplementedError(f"Unsupported number of channels: {nchannels}")
        
        image = pyglet.image.ImageData(
            self.width, 
            self.height, 
            _format, 
            arr.tobytes()
        )
        
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self) -> None:
        """Close the viewer window."""
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        """Clean up when viewer is garbage collected."""
        self.close()
