from hopfield.base import Hopfield
from hopfield.rules import HebbianRule, OjiRule
from hopfield.helpers import plot_pattern_evolution

from typing import cast
import numpy as np

pattern = np.genfromtxt("data/projekt2/small-7x7.csv", delimiter=",")[0]

model = Hopfield((7, 7), OjiRule(1, seed=858781142), 0)
model.train([pattern])

damaged_pattern = pattern
damaged_pattern[:40] = -1

_, history = model.predict(damaged_pattern, "asynchronous", save_history=True)
history = cast(list[np.ndarray], history)
plot_pattern_evolution(history, (7, 7))
