from abc import ABC, abstractmethod
import numpy as np


class LearningRule(ABC):
    @abstractmethod
    def __call__(self, neuron_amount: int, patterns: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        pass


class HebbianRule(LearningRule):
    def __call__(self, neuron_amount: int, patterns: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        weights = np.zeros((neuron_amount, neuron_amount))
        bias = np.zeros((neuron_amount, 1))
        for x in patterns:
            weights += np.outer(x, x) - np.eye(neuron_amount)
            bias += x

        return weights / len(patterns), bias / len(patterns)


class OjiRule(LearningRule):
    def __init__(self, lr: float):
        self.lr = lr

    def __call__(self, neuron_amount: int, patterns: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        weights = np.zeros((neuron_amount, neuron_amount))
        bias = np.zeros((neuron_amount, 1))

        for x in patterns:
            y = weights @ x
            weights += self.lr * (y @ (x.T - y.T @ weights))
            bias += x

        return weights / len(patterns), bias / len(patterns)
