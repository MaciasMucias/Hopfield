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
