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
        mask = ~np.eye(*weights_shape, dtype=bool)
        weights = np.abs(self.rng.standard_normal(size=weights_shape)) * 0.00001 * mask
        bias = np.zeros((neuron_amount,))

        effective_lr = self.lr / len(patterns)

        for x in patterns:
            outer_product = np.outer(x, x)
            dot_products = np.sum(x[:, np.newaxis] * x, axis=1)
            weights += effective_lr * (outer_product - weights * dot_products[np.newaxis, :]) * mask

            # weight_update = np.zeros(weights_shape)
            # for i in range(neuron_amount):
            #     for j in range(neuron_amount):
            #         if i == j:
            #             continue
            #         weight_update[i, j] = effective_lr * (np.dot(x[i], x[j]) - weights[i, j] * np.dot(x[j], x[j]))
            # weights += weight_update
            # y = weights.T @ x
            # if np.any(np.isnan(y)):
            #     raise RuntimeError("Weight explosion")
            # weights += effective_lr * np.outer(x - weights @ y, y) * mask
            bias += x

        return weights, bias / len(patterns)
