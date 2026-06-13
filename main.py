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
from src.storage.response_store import save_response

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

dataset = load_truthfulqa()

for i in range(30):

    sample = dataset[i]

    question = sample["question"]
    ground_truth = sample["best_answer"]

    #check cache first
    cached_response = get_cached_response(question)

    if cached_response is not None:
        print("Using cached response...")
        response = cached_response
    else:
        print("Querying Gemini...")
        response = ask_gemini(question)
        save_to_cache(question,response)
        time.sleep(3)
    
    #evaluate
    exact_result = exact_match(response,ground_truth)
    keyword_result = keyword_match(response,ground_truth)
    semantic_score = semantic_similarity(response,ground_truth)
    semantic_result = semantic_verdict(response,ground_truth)

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
        semantic_match=semantic_result
    )

