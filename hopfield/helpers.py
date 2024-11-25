import matplotlib.animation as anim
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from typing import Literal

from .base import Hopfield


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


def pattern_completion_error(output_pattern: np.ndarray, expected_pattern: np.ndarray, shape: tuple[int, int]) -> float:
    total_bits = expected_pattern.size
    wrong_bits = np.count_nonzero(expected_pattern - output_pattern)
    return wrong_bits / total_bits


def average_pattern_completion_error(
    model: Hopfield,
    patterns: list[np.ndarray],
    update_procedure: Literal["synchronous", "asynchronous"],
    noise_ratio: float,
    it_per_pattern: int = 100,
) -> list[float]:
    model.train(patterns)
    shape = (model.width, model.height)

    errors = []
    for pattern in patterns:
        model_error = 0
        for _ in range(it_per_pattern):
            destroyed_pattern = np.where(
                model.rng.choice([True, False], size=shape, p=[noise_ratio, 1 - noise_ratio]), 0, pattern
            )
            recovered_pattern = model.predict(destroyed_pattern, update_procedure, False)

            # pattern may be a negative, error should be 0 then
            # from this we can reason that the max error should be 0.5
            raw_error = pattern_completion_error(recovered_pattern, pattern)

            # Map error from key points 0, 0.5, 1 -> 0, 0.5, 0
            model_error += -(abs(raw_error - 1 / 2) - 1 / 2)

        errors.append(model_error / it_per_pattern)

    return errors
