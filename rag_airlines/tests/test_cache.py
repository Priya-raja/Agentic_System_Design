from context.memory.answer_cache import create_answer_cache_key
from context.memory.semantic_answer_cache import semantic_cache_allowed


def test_prompt_hash_invalidates_exact_cache_and_unsafe_queries_bypass_semantic():
    assert create_answer_cache_key("same question", "v1") != create_answer_cache_key("same question", "v2")
    for question in ["What is my refund?", "Compare baggage and upgrades", "What is allowed tomorrow?"]:
        assert not semantic_cache_allowed(question, "baggage")
