import sys
import os
# Ensure project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from src.visualization.charts import (
    evaluator_comparison_chart,
    threshold_chart
)
from src.scoring.disagreement_analytics import (
    calculate_disagreement_stats,
    generate_disagreement_chart
)
from src.reports.report_generator import build_pdf_report

FILE_PATH = "results/csv/evaluation_results.csv"
VIS_DIR = "results/visualizations"
REPORTS_DIR = "results/reports"


def load_results():
    return pd.read_csv(FILE_PATH)
def calculate_metrics(df):

    total = len(df)

    exact_acc = (
        df["exact_match"].sum() / total
    ) * 100

    keyword_acc = (
        df["keyword_match"].sum() / total
    ) * 100

    semantic_acc = (
        df["semantic_match"].sum() / total
    ) * 100

    avg_semantic = (
        df["semantic_score"].mean()
    )
    judge_acc = 0

    if "judge_score" in df.columns:
        judge_acc = (
            df["judge_score"].fillna(0).sum()
            / total
        ) * 100

    return {
        "total_questions": total,
        "exact_accuracy": exact_acc,
        "keyword_accuracy": keyword_acc,
        "semantic_accuracy": semantic_acc,
        "average_semantic_score": avg_semantic,
        "judge_accuracy": judge_acc
    }

def print_report(metrics):

    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)

    print(
        f"Total Questions: {metrics['total_questions']}"
    )

    print(
        f"Exact Match Accuracy: {metrics['exact_accuracy']:.2f}%"
    )

    print(
        f"Keyword Match Accuracy: {metrics['keyword_accuracy']:.2f}%"
    )

    print(
        f"Semantic Match Accuracy: {metrics['semantic_accuracy']:.2f}%"
    )

    print(
        f"Average Semantic Score: {metrics['average_semantic_score']:.3f}"
    )

    print(
        f"Judge Accuracy: {metrics['judge_accuracy']:.2f}%"
    )

def category_analysis(df):

    category_stats = (
        df.groupby("category")
        ["semantic_match"]
        .mean()
        * 100
    )

    print("\nCATEGORY ANALYSIS")
    print("-" * 30)

    for category, accuracy in category_stats.items():

        print(
            f"{category}: {accuracy:.2f}%"
        )

def failure_analysis(df):

    failures = df[
        df["semantic_match"] == False
    ]

    print("\nFAILURE ANALYSIS")
    print("-" * 30)

    if failures.empty:
        print("No failures found.")
        return

    for _, row in failures.iterrows():

        print("\nQUESTION:")
        print(row["question"])

        print("\nGROUND TRUTH:")
        print(row["ground_truth"])

        print("\nRESPONSE:")
        print(row["response"])

        print("\nSEMANTIC SCORE:")
        print(round(row["semantic_score"], 3))

        print("\n" + "=" * 80)

def evaluator_summary(metrics):

    print("\nEVALUATOR COMPARISON")
    print("-" * 40)

    print(f"Exact Match      : {metrics['exact_accuracy']:.2f}%")
    print(f"Keyword Match    : {metrics['keyword_accuracy']:.2f}%")
    print(f"Semantic Match   : {metrics['semantic_accuracy']:.2f}%")
    print(f"Gemini Judge Match      : {metrics['judge_accuracy']:.2f}%")

def threshold_experiment(df):

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    accuracies = []

    print("\nTHRESHOLD ANALYSIS")
    print("-" * 40)

    for threshold in thresholds:

        accuracy = (
            (df["semantic_score"] >= threshold)
            .mean()
            * 100
        )
        accuracies.append(accuracy)

        print(
            f"Threshold {threshold:.2f} -> "
            f"{accuracy:.2f}%"
        )

    threshold_chart(thresholds, accuracies)

def judge_disagreement_analysis(df, threshold=0.6):

    if "judge_score" not in df.columns:
        print("\n[WARNING] Gemini Judge score not present in dataset. Skipping disagreement analysis.")
        return

    stats = calculate_disagreement_stats(df, threshold)
    if not stats:
        print("\n[WARNING] No valid Gemini Judge evaluations found. Skipping disagreement analysis.")
        return

    print("\nJUDGE DISAGREEMENT ANALYTICS")
    print("-" * 40)
    print(f"Total Evaluated Questions   : {stats['total_evaluated']}")
    print(f"Agreement Rate              : {stats['agreement_rate']:.2f}% ({stats['total_agreed_count']} cases)")
    print(f"Disagreement Rate            : {stats['disagreement_rate']:.2f}% ({stats['total_disagreed_count']} cases)")
    print(f" - False Positives (Sem Pass, Judge Fail) : {stats['false_positive_count']} cases ({stats['false_positive_rate']:.2f}%)")
    print(f" - False Negatives (Sem Fail, Judge Pass) : {stats['false_negative_count']} cases ({stats['false_negative_rate']:.2f}%)")
    print(f"Keyword Disagreement Rate   : {stats['keyword_disagreement_rate']:.2f}% ({stats['keyword_disagreed_count']} cases)")
    print(f" - Keyword Fallacies (KW Pass, Judge Fail): {stats['keyword_fallacy_count']} cases")
    print(f" - Keyword Omissions (KW Fail, Judge Pass): {stats['keyword_omission_count']} cases")

    # Generate disagreement chart
    generate_disagreement_chart(stats, VIS_DIR)

    # Generate PDF report
    pdf_path = os.path.join(REPORTS_DIR, "evaluation_summary.pdf")
    build_pdf_report(FILE_PATH, pdf_path, threshold, VIS_DIR)


if __name__ == "__main__":

    df = load_results()

    metrics = calculate_metrics(df)

    print_report(metrics)
    category_analysis(df)
    failure_analysis(df)
    evaluator_summary(metrics)
    threshold_experiment(df)
    evaluator_comparison_chart(metrics)
    judge_disagreement_analysis(df, threshold=0.6)