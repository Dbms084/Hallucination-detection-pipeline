from asyncio import timeouts
import sys
import time
from src.dataset.fetch_dataset import load_truthfulqa
from src.models.query_gemini import ask_gemini
from src.evaluators.exact_match import exact_match
from src.cache.cache_manager import (
    get_cached_response,
    save_to_cache
)
from src.evaluators.keyword_match import keyword_match
from src.evaluators.semantic_similarity import (
    semantic_similarity,
    semantic_verdict
)
from src.storage.response_store import save_response, load_responses
from src.evaluators.llm_judge import (
    evaluate_with_judge
)
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

dataset = load_truthfulqa()
existing_evals = load_responses()
quota_exhausted = False

for i in range(30):

    sample = dataset[i]

    question = sample["question"]
    ground_truth = sample["best_answer"]
    
    question_clean = question.replace("\r\n", "\\n").replace("\n", "\\n")

    # Check if this question is already fully evaluated (has judge_score and response)
    is_fully_evaluated = False
    if not existing_evals.empty and "question" in existing_evals.columns:
        match = existing_evals[existing_evals["question"] == question_clean]
        if not match.empty:
            row = match.iloc[0]
            # Ensure it contains a valid non-null judge_score and reason
            if "judge_score" in row and pd.notna(row["judge_score"]) and "judge_reason" in row and pd.notna(row["judge_reason"]):
                is_fully_evaluated = True

    if is_fully_evaluated:
        print(f"Question {i+1} already fully evaluated. Skipping...")
        continue

    #check cache first
    cached_response = get_cached_response(question)

    if cached_response is not None:
        print("Using cached response...")
        response = cached_response
    else:
        if quota_exhausted:
            print(f"\n[ERROR] Question {i+1}: Skipped response generation since Gemini API daily quota is exhausted.")
            break
        print("Querying Gemini...")
        try:
            response = ask_gemini(question)
            save_to_cache(question,response)
            time.sleep(3)
        except Exception as e:
            is_quota_error = "429" in str(e) or "quota" in str(e).lower() or "resourceexhausted" in str(type(e)).lower()
            if is_quota_error:
                print(f"\n[ERROR] Gemini API quota exceeded during response generation: {e}")
                print("Exiting generation loop gracefully. Existing evaluations remain saved.")
                break
            else:
                raise e
    
    #evaluate
    exact_result = exact_match(response,ground_truth)
    keyword_result = keyword_match(response,ground_truth)
    semantic_score = semantic_similarity(response,ground_truth)
    semantic_result = semantic_verdict(response,ground_truth)
    
    if quota_exhausted:
        judge_score = None
        judge_reason = "Daily Gemini quota exceeded; evaluation skipped"
    else:
        try:
            judge_result = evaluate_with_judge(
                question,
                ground_truth,
                response
            )
            judge_score = judge_result["judge_score"]
            judge_reason = judge_result["judge_reason"]
        except Exception as e:
            is_quota_error = "429" in str(e) or "quota" in str(e).lower() or "resourceexhausted" in str(type(e)).lower()
            if is_quota_error:
                print(f"\n[ERROR] Gemini Judge API quota exceeded: {e}")
                print("Skipping judge evaluation for this and remaining questions in this run. Other metrics will still be computed and saved.")
                quota_exhausted = True
                judge_score = None
                judge_reason = "Daily Gemini quota exceeded; evaluation skipped"
            else:
                raise e

    print(f"\nQuestion {i+1}")
    print(question)

    print("\nGround Truth:")
    print(ground_truth)

    print("\nGemini:")
    print(response)

    print("\nExact Match:")
    print(exact_result)

    print("\nKeyword Match: ")
    print(keyword_result)

    print("\nSemantic Score: ")
    print(round(semantic_score,3))

    print("\nSemantic Verdict:")
    print(semantic_result)

    # Save evaluation results to CSV
    save_response(
        question=question,
        category=sample.get("category", "N/A"),
        ground_truth=ground_truth,
        response=response,
        exact_match=exact_result,
        keyword_match=keyword_result,
        semantic_score=semantic_score,
        semantic_match=semantic_result,
        judge_score=judge_score,
        judge_reason=judge_reason
    )
    time.sleep(2) # Prevent rate-limiting during evaluation

