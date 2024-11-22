from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class LearningRule(ABC):
    @abstractmethod
    def __call__(self, neuron_amount: int, patterns: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        pass


class HebbianRule(LearningRule):
    def __call__(self, neuron_amount: int, patterns: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        weights = np.zeros((neuron_amount, neuron_amount))
        bias = np.zeros((neuron_amount,))
        for x in patterns:
            weights += np.outer(x, x) - np.eye(neuron_amount)
            bias += x

        return weights / len(patterns), bias / len(patterns)


class OjiRule(LearningRule):
    def __init__(self, lr: float, *, seed: Optional[int] = None):
        self.lr = lr
        if seed is None:
            seed = np.random.randint(0, 2**31 - 1)
            print(f"{seed=}")
        self.rng = np.random.default_rng(seed)

    def __call__(self, neuron_amount: int, patterns: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        weights_shape = (neuron_amount, neuron_amount)
        mask = np.ones(weights_shape) - np.eye(*weights_shape)
        weights = self.rng.standard_normal(size=weights_shape) * 0.001 * mask
        bias = np.zeros((neuron_amount,))

        for x in patterns:
            y = weights.T @ x
            weights += self.lr * np.outer(x - weights @ y, y) * mask
            bias += x

        return weights / len(patterns), bias / len(patterns)
