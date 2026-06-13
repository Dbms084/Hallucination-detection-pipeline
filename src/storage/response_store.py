import pandas as pd
import os


RESULT_FILE = "results/csv/evaluation_results.csv"


def save_response(
    question,
    category,
    ground_truth,
    response,
    exact_match,
    keyword_match,
    semantic_score=None,
    semantic_match=None
):
    """
    Save one evaluation result to evaluation_results.csv
    """

    # Escape newlines to keep each entry on a single line in the CSV
    question_clean = str(question).replace("\r\n", "\\n").replace("\n", "\\n")
    ground_truth_clean = str(ground_truth).replace("\r\n", "\\n").replace("\n", "\\n")
    response_clean = str(response).replace("\r\n", "\\n").replace("\n", "\\n")

    new_row = pd.DataFrame({
        "question": [question_clean],
        "category": [category],
        "ground_truth": [ground_truth_clean],
        "response": [response_clean],
        "exact_match": [exact_match],
        "keyword_match": [keyword_match],
        "semantic_score": [semantic_score],
        "semantic_match": [semantic_match]
    })

    if os.path.exists(RESULT_FILE) and os.path.getsize(RESULT_FILE) > 0:

        existing_df = pd.read_csv(RESULT_FILE)

        # Prevent duplicate entries
        duplicate = existing_df[
            existing_df["question"] == question
        ]

        if not duplicate.empty:
            print("Question already exists. Skipping save.")
            return

        updated_df = pd.concat(
            [existing_df, new_row],
            ignore_index=True
        )

    else:
        updated_df = new_row

    # Ensure parent directory exists before saving
    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    updated_df.to_csv(
        RESULT_FILE,
        index=False
    )

    print("Response saved successfully.")


def load_responses():
    """
    Load evaluation results from CSV
    """

    if os.path.exists(RESULT_FILE) and os.path.getsize(RESULT_FILE) > 0:
        return pd.read_csv(RESULT_FILE)

    return pd.DataFrame()


if __name__ == "__main__":

    save_response(
        question="Who invented Python?",
        category="Technology",
        ground_truth="Guido van Rossum",
        response="Python was created by Guido van Rossum.",
        exact_match=False,
        keyword_match=True
    )

    df = load_responses()

    print(df.head())