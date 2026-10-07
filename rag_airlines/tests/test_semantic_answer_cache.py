import json
import time
import unittest
from unittest.mock import MagicMock

from context.memory.semantic_answer_cache import (
    SemanticAnswerCache, question_signature, semantic_cache_allowed,
)
from config import ANSWER_CACHE_TTL_SECONDS


class SemanticCacheTests(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.embeddings = MagicMock()
        self.embeddings.embed_query.return_value = [1.0, 0.0]
        self.cache = SemanticAnswerCache(self.embeddings, self.client)
        self.scope = dict(topic='baggage', policy_year=2026, prompt_version='v4',
                          prompt_hash='hash', corpus_version='corpus')
        self.question = 'What is the checked baggage allowance for Economy Flex in 2026?'
        self.payload = {'response': {'answerable': True}, 'results': []}
        self.entry = dict(created_at=time.time(), signature=question_signature(self.question),
                          embedding=[1.0, 0.0], payload=self.payload)
        self.client.lrange.return_value = [json.dumps(self.entry)]

    def test_paraphrase_hit(self):
        question = 'Please tell me the checked luggage allowance for Economy Flex in 2026.'
        self.assertEqual(self.cache.lookup(question=question, **self.scope), self.payload)

    def test_changed_fare_year_route_or_quantity_miss(self):
        for question in [self.question.replace('Flex', 'Saver'),
                         self.question.replace('2026', '2025'),
                         self.question + ' DXB LHR', self.question + ' 30 kg']:
            self.assertIsNone(self.cache.lookup(question=question, **self.scope))
        self.embeddings.embed_query.assert_not_called()

    def test_low_similarity_miss(self):
        self.embeddings.embed_query.return_value = [0.0, 1.0]
        self.assertIsNone(self.cache.lookup(question=self.question, **self.scope))

    def test_expired_entry_miss(self):
        self.entry['created_at'] = time.time() - ANSWER_CACHE_TTL_SECONDS - 1
        self.client.lrange.return_value = [json.dumps(self.entry)]
        self.assertIsNone(self.cache.lookup(question=self.question, **self.scope))

    def test_each_scope_change_invalidates(self):
        original = self.cache._key(self.scope)
        for name in self.scope:
            scope = {**self.scope, name: str(self.scope[name]) + '-changed'}
            self.assertNotEqual(original, self.cache._key(scope))

    def test_failures_fall_back(self):
        self.client.lrange.side_effect = RuntimeError('redis unavailable')
        self.assertIsNone(self.cache.lookup(question=self.question, **self.scope))

    def test_refusal_not_stored(self):
        self.cache.store(question=self.question, answer={'response': {'answerable': False}}, metadata=self.scope)
        self.client.pipeline.assert_not_called()

    def test_store_is_bounded_and_expiring(self):
        self.cache.store(question=self.question, answer=self.payload, metadata=self.scope)
        pipeline = self.client.pipeline.return_value.__enter__.return_value
        pipeline.lpush.assert_called_once()
        pipeline.ltrim.assert_called_once()
        pipeline.expire.assert_called_once()
        pipeline.execute.assert_called_once()

    def test_routing_and_sensitive_questions_bypass(self):
        topic = 'baggage'
        self.assertTrue(semantic_cache_allowed(self.question, topic))
        for question in ['What is my baggage allowance?', 'Compare Flex and Saver',
                         'What is allowed today?', 'Are power banks not permitted?']:
            self.assertFalse(semantic_cache_allowed(question, topic))
        self.assertFalse(semantic_cache_allowed(self.question, None))

    def test_route_direction_remains_distinct(self):
        self.assertNotEqual(question_signature('menu from DXB to LHR'),
                            question_signature('menu from LHR to DXB'))

    def test_limit_and_inclusion_remain_distinct(self):
        self.assertNotEqual(question_signature('maximum baggage weight'),
                            question_signature('included baggage weight'))


if __name__ == '__main__':
    unittest.main()
