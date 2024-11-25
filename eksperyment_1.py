from typing import Sequence, TypedDict
import numpy as np
import matplotlib.pyplot as plt

from hopfield.base import Hopfield
from hopfield.rules import OjiRule, HebbianRule


class EvalConfig(TypedDict):
    shape: tuple[int, int]
    noise_ratio: float
    iterations: int
    seed: int


def evaluate_hebbian(config: EvalConfig, patterns: Sequence[np.ndarray]) -> dict:
    model = Hopfield(config["shape"], HebbianRule(), config["seed"])
    errors = evaluate_model(model, patterns, config)
    return {"errors": errors, "mean_error": np.mean(errors)}


def evaluate_oja(config: EvalConfig, patterns: Sequence[np.ndarray], learning_rates: np.ndarray) -> dict:
    results = []
    for lr in learning_rates:
        model = Hopfield(config["shape"], OjiRule(lr=lr), config["seed"])
        errors = evaluate_model(model, patterns, config)
        results.append({"lr": lr, "errors": errors, "mean_error": np.mean(errors)})

    best_result = min(results, key=lambda x: x["mean_error"])
    return {
        "learning_rates": learning_rates,
        "results": results,
        "optimal_lr": best_result["lr"],
        "min_error": best_result["mean_error"],
    }


def evaluate_model(model: Hopfield, patterns: Sequence[np.ndarray], config: EvalConfig) -> np.ndarray:
    return np.array([evaluate_pattern(model, pattern, config["noise_ratio"]) for pattern in patterns])


def evaluate_pattern(model: Hopfield, pattern: np.ndarray, noise_ratio: float) -> float:
    model.train([pattern])
    corrupted = pattern.copy()
    mask = np.random.random(pattern.shape) < noise_ratio
    corrupted[mask] *= -1
    result, _ = model.predict(corrupted, "synchronous", save_history=False)
    error = min(np.mean(result != pattern), np.mean(result != -pattern))
    return error


def plot_learning_rate_curve(results: dict) -> None:
    plt.figure(figsize=(10, 6))
    plt.semilogx(results["learning_rates"], [r["mean_error"] for r in results["results"]], "b-", label="Error curve")
    plt.plot(results["optimal_lr"], results["min_error"], "ro", label=f"Minimum (lr={results['optimal_lr']:.4f})")
    plt.grid(True)
    plt.xlabel("Learning rate")
    plt.ylabel("Mean error")
    plt.legend()
    plt.title("Learning Rate vs Error for Oja's Rule")
    plt.show()


def plot_rule_comparison(hebbian_results: dict, oja_results: dict, pattern_count: int) -> None:
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(pattern_count) - 0.2, hebbian_results["errors"], 0.4, label="Hebbian", color="blue", alpha=0.6)
    best_oja = min(oja_results["results"], key=lambda x: x["mean_error"])
    plt.bar(
        np.arange(pattern_count) + 0.2,
        best_oja["errors"],
        0.4,
        label=f"Oja (lr={best_oja['lr']:.4f})",
        color="red",
        alpha=0.6,
    )
    plt.xlabel("Pattern Index")
    plt.ylabel("Error Rate")
    plt.title("Pattern-wise Error Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_letter_patterns(model: Hopfield, patterns: list[np.ndarray], config: EvalConfig) -> None:
    import string

    rows, cols = 5, 6
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))

    for i, (pattern, letter) in enumerate(zip(patterns, string.ascii_uppercase)):
        row, col = i // cols, i % cols
        ax = axes[row, col]

        # Create corrupted pattern
        corrupted = pattern.copy()
        mask = np.random.random(pattern.shape) < config["noise_ratio"]
        corrupted[mask] *= -1

        # Get prediction
        result, _ = model.predict(corrupted, "synchronous", save_history=False)
        error = min(np.mean(result != pattern), np.mean(result != -pattern)) / 2

        # Show original, corrupted and predicted side by side
        combined = np.hstack(
            [pattern.reshape(config["shape"]), corrupted.reshape(config["shape"]), result.reshape(config["shape"])]
        )

        ax.imshow(combined, cmap="binary")
        ax.set_title(f"Letter {letter}\nError: {error:.2f}")
        ax.axis("off")

    # Clear unused subplots
    for i in range(len(string.ascii_uppercase), rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis("off")

    plt.suptitle("Original vs Corrupted vs Predicted Patterns")
    plt.tight_layout()
    plt.show()


def main():
    patterns = np.genfromtxt("data/projekt2/letters-14x20.csv", delimiter=",")
    config: EvalConfig = {"shape": (20, 14), "noise_ratio": 0.1, "iterations": 1, "seed": 0}

    hebbian_results = evaluate_hebbian(config, patterns)
    oja_results = evaluate_oja(config, patterns, np.logspace(-4, -1, 50))

    plot_learning_rate_curve(oja_results)
    plot_rule_comparison(hebbian_results, oja_results, len(patterns))

    hebbian_model = Hopfield(config["shape"], HebbianRule(), config["seed"])
    hebbian_model.train(patterns)

    oja_model = Hopfield(config["shape"], OjiRule(lr=0.4), config["seed"])
    oja_model.train(patterns)

    visualize_letter_patterns(hebbian_model, patterns, config)
    visualize_letter_patterns(oja_model, patterns, config)


if __name__ == "__main__":
    main()
