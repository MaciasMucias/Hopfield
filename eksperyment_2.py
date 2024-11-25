from typing import cast
import numpy as np
from hopfield.base import Hopfield
from hopfield.rules import HebbianRule, OjiRule
from hopfield.helpers import first_last_frame

pattern = list(np.genfromtxt("Projekt2/new_patterns.csv", delimiter=","))

model = Hopfield((100, 100), HebbianRule(), 0)
model.train(pattern)

# Uszkodzenia 25%
damaged_pattern = pattern[2]
damaged_pattern[:2500] = -1

_, history = model.predict(damaged_pattern, "synchronous", save_history=True)
history = cast(list[np.ndarray], history)
first_last_frame(history, (100, 100))

# Uszkodzenia 50%
damaged_pattern = pattern[2]
damaged_pattern[:5000] = -1

_, history = model.predict(damaged_pattern, "synchronous", save_history=True)
history = cast(list[np.ndarray], history)
first_last_frame(history, (100, 100))

# Uszkodzenia 75%
damaged_pattern = pattern[2]
damaged_pattern[:7500] = -1

_, history = model.predict(damaged_pattern, "synchronous", save_history=True)
history = cast(list[np.ndarray], history)
first_last_frame(history, (100, 100))

# Uszkodzenia 90%
damaged_pattern = pattern[2]
damaged_pattern[:9000] = -1

_, history = model.predict(damaged_pattern, "synchronous", save_history=True)
history = cast(list[np.ndarray], history)
first_last_frame(history, (100, 100))
