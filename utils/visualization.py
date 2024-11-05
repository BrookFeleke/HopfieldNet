# utils/visualization.py

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_predictions(states, sync: bool):
    """
    Plots the predictions of a Hopfield network for each pattern and saves the figure.

    Args:
        states: A list of lists containing the states of each pattern at different stages.
                Each inner list contains three states: original, noisy input, and predicted state.
        sync: A boolean indicating whether the prediction was synchronous or asynchronous.

    The function creates a grid of subplots where each row corresponds to a pattern and each column
    represents a stage (original, noisy input, predicted state). The plot is saved as a PNG file
    with a filename that indicates whether the prediction was synchronous or asynchronous.
    """
    n_patterns = len(states)
    n_transitions = len(states[0])
    fig, ax = plt.subplots(n_patterns, n_transitions, figsize=(10, 5))
    for j in range(n_patterns):
        for k in range(n_transitions):
            ax[j, k].matshow(states[j][k].reshape((28, 28)), cmap='gray')
            ax[j, k].set_xticks([])
            ax[j, k].set_xticks([])
            ax[j, k].set_xticks([])
            ax[j, k].set_yticks([])
            ax[j, k].set_yticks([])
            ax[j, k].set_yticks([])
            title = 'Original' if k == 0 else 'Noisy Input' if k == 1 else 'Predicted State'
            ax[j, k].set_title(f'{title}_{j}')
    prefix = 'sync' if sync else 'async'
    save_path = Path('./png') / f'{prefix}_prediction'
    plt.savefig(save_path)
    plt.close()

def plot_energy_transition(energies, sync: bool):
    x_axis = np.arange(len(energies[0]))
    for energy_idx in range(len(energies)):
        plt.plot(x_axis, energies[energy_idx], label=f'pattern_{energy_idx}')
    plt.legend()
    prefix = 'sync' if sync else 'async'
    save_path = Path('./png') / f'{prefix}_energy_transition'
    plt.savefig(save_path)
    plt.close()
