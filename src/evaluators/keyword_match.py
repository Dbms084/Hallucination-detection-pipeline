def keyword_match(response, answer):

    response = response.lower()
    answer = answer.lower().split()

    important_words = answer

    matches = 0

    for word in important_words:
        if word in response:
            matches += 1

    score = matches / len(important_words)

    return score >= 0.6