from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What percentage of the brain does a human typically use?",
    actual_output="""
    Humans use 100% of their brain. The 10% myth is false.
    """
)

metric = AnswerRelevancyMetric(
    threshold=0.7
)

metric.measure(test_case)

print("Score:", metric.score)
print("Passed:", metric.success)
print("Reason:", metric.reason)