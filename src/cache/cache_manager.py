import pandas as pd
import os

CACHE_FILE = "results/cache/responses.csv"


def load_cache():
    if os.path.exists(CACHE_FILE) and os.path.getsize(CACHE_FILE) > 0:
        return pd.read_csv(CACHE_FILE)

    return pd.DataFrame(columns=["question", "response"])


def save_to_cache(question, response):

    df = load_cache()

    new_row = pd.DataFrame({
        "question": [question],
        "response": [response]
    })

    df = pd.concat([df, new_row], ignore_index=True)

    # Ensure parent directory exists before saving
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    df.to_csv(CACHE_FILE, index=False)

def get_cached_response(question):

    df = load_cache()

    match = df[df["question"] == question]

    if not match.empty:
        return match.iloc[0]["response"]

    return None