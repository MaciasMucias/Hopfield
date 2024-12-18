import numpy as np
from typing import Literal, Optional, cast
from copy import copy

from .rules import LearningRule, NondeterministicLearningRule


class Hopfield:
    def __init__(self, input_shape: tuple[int, int], learning_rule: LearningRule, seed: Optional[int] = None) -> None:
        self.width, self.height = input_shape
        self.total_neurons = self.width * self.height
        self.learning_rule = learning_rule

        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
            print(f"{seed=}")

        self.rng = np.random.default_rng(seed)

        if not self.learning_rule.is_deterministic:
            self.learning_rule = cast(NondeterministicLearningRule, self.learning_rule)
            self.learning_rule.set_rng(self.rng)

    def train(self, patterns: list[np.ndarray]) -> None:
        self.weights, self.bias = self.learning_rule(self.total_neurons, patterns)

    def get_new_pattern_synchronous(self, pattern: np.ndarray) -> np.ndarray:
        return 2 * (np.dot(self.weights, pattern) >= 0).astype(int) - 1

    def get_new_pattern_asynchronous(self, pattern: np.ndarray) -> np.ndarray:
        neuron_idx = list(range(self.total_neurons))
        self.rng.shuffle(neuron_idx)
        for i in neuron_idx:
            pattern[i] = 2 * (np.dot(self.weights[i], pattern) >= 0).astype(int) - 1
        return pattern

    def predict(
        self, pattern: np.ndarray, update_procedure: Literal["synchronous", "asynchronous"], *, save_history: bool
    ) -> tuple[np.ndarray, Optional[list[np.ndarray]]]:
        if save_history:
            patterns_history = [pattern]
        else:
            patterns_history = None
        encountered_patterns = set()
        encountered_patterns.add(hash(pattern.tobytes()))
        while True:
            if update_procedure == "synchronous":
                pattern = self.get_new_pattern_synchronous(pattern)
            elif update_procedure == "asynchronous":
                pattern = self.get_new_pattern_asynchronous(copy(pattern))
            else:
                raise ValueError(f"update_procedure does not support value: {update_procedure}")

            pattern_hash = hash(pattern.tobytes())
            if save_history:
                patterns_history = cast(list[np.ndarray], patterns_history)
                patterns_history.append(pattern)
            if pattern_hash in encountered_patterns:
                break
            encountered_patterns.add(pattern_hash)

        return pattern, patterns_history
