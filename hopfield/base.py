import numpy as np
from typing import Literal, Optional

from .rules import LearningRule


class Hopfield:
    def __init__(self, input_shape: tuple[int, int], learning_rule: LearningRule, seed: Optional[int] = None) -> None:
        self.width, self.height = input_shape
        self.total_neurons = self.width * self.height
        self.learning_rule = learning_rule

        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
            print(f"{seed=}")

        self.rng = np.random.default_rng(seed)

    def train(self, patterns: list[np.ndarray]):
        self.weights, self.bias = self.learning_rule(self.total_neurons, patterns)

    def get_new_pattern_synchronous(self, pattern: np.ndarray) -> np.ndarray:
        return np.sign(np.dot(self.weights, pattern))

    def get_new_pattern_asynchronous(self, pattern: np.ndarray) -> np.ndarray:
        neuron_idx = list(range(self.total_neurons))
        self.rng.shuffle(neuron_idx)
        for i in neuron_idx:
            pattern[i] = np.sign(np.dot(self.weights[i], pattern))
        return pattern

    def predict(self, pattern: np.ndarray, update_procedure: Literal["synchronous", "asynchronous"]):
        patterns_history = [pattern]
        encountered_patterns = set()
        encountered_patterns.add(hash(pattern))
        while True:
            if update_procedure == "synchronous":
                pattern = self.get_new_pattern_synchronous(pattern)
            elif update_procedure == "asynchronous":
                pattern = self.get_new_pattern_asynchronous(pattern)
            else:
                raise ValueError(f"update_procedure does not support value: {update_procedure}")

            pattern_hash = hash(pattern)
            if pattern_hash in encountered_patterns:
                break
            patterns_history.append(pattern)
            encountered_patterns.add(pattern_hash)
