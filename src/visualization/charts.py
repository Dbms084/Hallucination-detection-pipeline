import matplotlib.pyplot as plt
import os

# Resolve directory paths absolutely based on script file location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIS_DIR = os.path.join(BASE_DIR, "results", "visualizations")

os.makedirs(
    VIS_DIR,
    exist_ok=True
)


def evaluator_comparison_chart(metrics):

    evaluators = [
        "Exact Match",
        "Keyword Match",
        "Semantic Match",
        "Gemini Judge"
    ]

    accuracies = [
        metrics["exact_accuracy"],
        metrics["keyword_accuracy"],
        metrics["semantic_accuracy"],
        metrics["judge_accuracy"]
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        evaluators,
        accuracies
    )

    plt.title("Evaluator Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Evaluator")

    plt.tight_layout()

    plt.savefig(
        os.path.join(VIS_DIR, "evaluator_comparison.png")
    )

    plt.close()

    print(
        "Saved evaluator_comparison.png"
    )

def threshold_chart(thresholds=None, accuracies=None):

    if thresholds is None:
        thresholds = [
            0.50,
            0.55,
            0.60,
            0.65,
            0.70
        ]

    if accuracies is None:
        accuracies = [
            100,
            87.5,
            87.5,
            62.5,
            50
        ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        thresholds,
        accuracies,
        marker="o"
    )

    plt.title(
        "Threshold vs Accuracy"
    )

    plt.xlabel(
        "Semantic Threshold"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(VIS_DIR, "threshold_analysis.png")
    )

    plt.close()

    print(
        "Saved threshold_analysis.png"
    )

if __name__ == "__main__":
    # Test chart generation with mock metrics
    mock_metrics = {
        "exact_accuracy": 0.0,
        "keyword_accuracy": 100.0,
        "semantic_accuracy": 78.26
    }
    evaluator_comparison_chart(mock_metrics)
    threshold_chart()