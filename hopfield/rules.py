from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class LearningRule(ABC):
    is_deterministic: bool

    @abstractmethod
    def __call__(self, neuron_amount: int, patterns: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        pass


class NondeterministicLearningRule(LearningRule):
    @abstractmethod
    def set_rng(self, rng: np.random.Generator) -> None:
        pass


class HebbianRule(LearningRule):
    is_deterministic = True

    def __call__(self, neuron_amount: int, patterns: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        weights = np.zeros((neuron_amount, neuron_amount))
        bias = np.zeros((neuron_amount,))
        for x in patterns:
            weights += np.outer(x, x) - np.eye(neuron_amount)
            bias += x

        return weights / len(patterns), bias / len(patterns)


class OjiRule(NondeterministicLearningRule):
    is_deterministic = False

    def __init__(self, lr: float) -> None:
        self.lr = lr

    def set_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng

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
