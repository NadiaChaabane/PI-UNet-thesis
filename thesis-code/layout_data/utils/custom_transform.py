import torch
import numpy as np
import torchvision.transforms.functional as TF
import cv2



# In np_transforms.py

import torch

class AddGaussianNoise:
    """
    Additive Gaussian noise transform for torch tensors.

    Args:
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    """

    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std == 0.0:
            return x
        noise = torch.randn_like(x) * self.std + self.mean
        return x + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"




class GaussianBlur:
    """Apply Gaussian blur to a numpy array.
    Args:
        kernel_size (int): Size of the Gaussian kernel. Must be odd.
        sigma (float): Standard deviation of the Gaussian kernel.
    """

    def __init__(self, kernel_size=5, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, x):
        # If blur is disabled (sigma=0 or kernel=0), return input unchanged
        if self.kernel_size <= 0 or self.sigma <= 0.0:
            return x
        
        # Apply Gaussian blur to the 2D array
        return cv2.GaussianBlur(x, (self.kernel_size, self.kernel_size), self.sigma)


