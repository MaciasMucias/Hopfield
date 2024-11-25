import matplotlib.animation as anim
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def plot_pattern_evolution(patterns: list[np.ndarray], shape: tuple[int, int]) -> None:
    patterns = list(map(lambda x: ((x + 1) / 2).reshape(shape), patterns))
    frames = len(patterns)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    cax = ax.matshow(patterns[0], cmap="binary")
    plt.axis("off")

    ax_slider = plt.axes((0.2, 0.05, 0.6, 0.03), facecolor="lightgrey")
    slider = Slider(ax_slider, "Frame", 0, frames - 1, valinit=0, valstep=1)

    def update(val):
        frame = int(slider.val)
        cax.set_data(patterns[frame])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


def first_last_frame(patterns: list[np.ndarray], shape: tuple[int, int]) -> None:

    if len(patterns) > 1:
        patterns = [patterns[0], patterns[-1]]
    else:
        patterns = [patterns[0]]

    patterns = [((x + 1) / 2).reshape(shape) for x in patterns]

    frames = len(patterns)
    fig_height = 2 * frames
    fig, axes = plt.subplots(nrows=frames, ncols=1, figsize=(6, fig_height))

    if frames == 1:
        axes = [axes]

    for idx, (pattern, ax) in enumerate(zip(patterns, axes)):
        cax = ax.matshow(pattern, cmap="binary")
        ax.set_title(f"{'First' if idx == 0 else 'Last'} Frame")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_pattern_evolution(output_pattern: np.ndarray, expected_pattern: np.ndarray, shape: tuple[int, int]) -> float:
    total_bits = expected_pattern.size
    wrong_bits = np.count_nonzero(expected_pattern - output_pattern)
    return wrong_bits / total_bits
