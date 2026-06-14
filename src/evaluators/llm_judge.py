import re
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)


JUDGE_PROMPT = """
You are an expert evaluator.

Question:
{question}

Ground Truth Answer:
{ground_truth}

Model Response:
{response}

Determine whether the model response is factually correct
with respect to the ground truth.

Rules:
- Ignore wording differences.
- Focus on factual correctness.
- Accept longer explanations if they remain correct.
- Reject incorrect facts.

Return ONLY in this format:

SCORE: 1

REASON: one sentence

OR

SCORE: 0

REASON: one sentence
"""


from google.api_core.exceptions import ResourceExhausted
import time

def judge_response(
    question,
    ground_truth,
    response
):

    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        response=response
    )

    # Retry logic for rate limits (5 RPM free tier limit)
    max_retries = 5
    base_delay = 15
    for attempt in range(max_retries):
        try:
            result = model.generate_content(
                prompt
            )
            return result.text
        except (ResourceExhausted, Exception) as e:
            # Check for rate limit indicators in exception
            is_rate_limit = isinstance(e, ResourceExhausted) or "429" in str(e) or "quota" in str(e).lower()
            if is_rate_limit and attempt < max_retries - 1:
                # Check for daily limits
                is_daily_limit = "daily" in str(e).lower() or "limit: 20" in str(e) or "quota_id" in str(e).lower() or "RequestsPerDay" in str(e)
                if is_daily_limit:
                    print(f"\n[ERROR] Gemini Judge hit daily API quota limit. Raising error immediately.")
                    raise e
                print(f"\n[WARNING] Gemini Judge hit rate limit. Sleeping {base_delay} seconds before retry (Attempt {attempt+1}/{max_retries})...")
                time.sleep(base_delay)
            else:
                raise e


def parse_judge_response(
    judge_text
):

    score_match = re.search(
        r"SCORE:\s*(\d)",
        judge_text
    )

    reason_match = re.search(
        r"REASON:\s*(.*)",
        judge_text,
        re.DOTALL
    )

    score = (
        int(score_match.group(1))
        if score_match
        else 0
    )

    reason = (
        reason_match.group(1).strip()
        if reason_match
        else "No reason returned"
    )

    return score, reason


import json
import urllib.request
import urllib.error

def query_ollama(prompt, model_name="llama3.2:3b"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            resp_data = json.loads(response.read().decode("utf-8"))
            return resp_data.get("response", "")
    except Exception as e:
        print(f"\n[WARNING] Ollama connection failed: {e}. Make sure Ollama is running (`ollama serve`).")
        raise e

def evaluate_with_judge(
    question,
    ground_truth,
    response
):
    try:
        judge_output = judge_response(
            question,
            ground_truth,
            response
        )
        score, reason = parse_judge_response(
            judge_output
        )
        return {
            "judge_score": score,
            "judge_reason": reason
        }
    except Exception as e:
        # Check if it is a quota or API limit error
        is_api_error = "429" in str(e) or "quota" in str(e).lower() or "resourceexhausted" in str(type(e)).lower()
        if is_api_error:
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
            print(f"\n[WARNING] Gemini Judge API error. Falling back to local Ollama model ({ollama_model})...")
            try:
                prompt = JUDGE_PROMPT.format(
                    question=question,
                    ground_truth=ground_truth,
                    response=response
                )
                ollama_output = query_ollama(prompt, ollama_model)
                score, reason = parse_judge_response(ollama_output)
                return {
                    "judge_score": score,
                    "judge_reason": f"{reason} (Ollama local fallback)"
                }
            except Exception as ollama_err:
                print(f"[ERROR] Local Ollama fallback also failed: {ollama_err}")
                raise e
        else:
            raise e


if __name__ == "__main__":

    question = (
        "What percentage of the brain does a human typically use?"
    )

    ground_truth = (
        "The majority of a human's brain is almost always active"
    )

    response = (
        "Humans use 100 percent of their brain."
    )

    result = evaluate_with_judge(
        question,
        ground_truth,
        response
    )

    print(result)