from sentence_transformers import SentenceTransformer
from sentence_transformers import util

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def semantic_similarity(response, answer):

    response_embedding = model.encode(
        response,
        convert_to_tensor=True
    )

    answer_embedding = model.encode(
        answer,
        convert_to_tensor=True
    )

    similarity = util.cos_sim(
        response_embedding,
        answer_embedding
    )

    return float(similarity)

def semantic_verdict(
    response,
    answer,
    threshold=0.60
):

    score = semantic_similarity(
        response,
        answer
    )

    return score >= threshold