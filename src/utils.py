import random

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_spectrogram(spectrogram, figsize=(12, 3)):
    """
    Plot the given spectrogram as an image having frequency
    on the y-axis and time on the x-axis
    """
    # If a batch of spectrograms is given, select a random one
    if len(spectrogram.shape) > 2:
        if spectrogram.size(0) > 1:
            spectrogram = spectrogram[random.randint(0, len(spectrogram) - 1)]
        else:
            spectrogram = spectrogram.squeeze(0)

    # Convert from torch to numpy
    spectrogram = to_numpy(spectrogram.squeeze(0)).astype(np.float32)

    # Plot the spectrogram
    _, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(img, ax=ax)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def to_numpy(arr):
    """
    Convert the given array to the numpy format
    """
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return None
