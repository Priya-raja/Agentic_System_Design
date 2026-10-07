import json
import unittest
from unittest.mock import patch

from agent import factory
from agent.jev_classifier import JevRoutingDecision
from agent.prompt_manager import PromptVersion
from config import ANSWER_CACHE_TTL_SECONDS, NO_CONTEXT_CACHE_TTL_SECONDS
from context.memory import answer_cache


class NoContextCacheTests(unittest.TestCase):
    def test_redis_expiry_override_and_default(self):
        with patch.object(answer_cache, 'redis_client') as redis:
            answer_cache.set_cached_answer('no-context', {'message': 'missing'}, ttl_seconds=300)
            self.assertEqual(redis.set.call_args.kwargs['ex'], NO_CONTEXT_CACHE_TTL_SECONDS)
            self.assertEqual(json.loads(redis.set.call_args.args[1]), {'message': 'missing'})
            answer_cache.set_cached_answer('answer', {'response': {}})
            self.assertEqual(redis.set.call_args.kwargs['ex'], ANSWER_CACHE_TTL_SECONDS)

    def test_repeated_no_context_skips_routing_retrieval_and_generation(self):
        for question, message in [
            ('Was Economy Flex eligible for a Business upgrade in 2025?',
             'No applicable policy was found for 2025.'),
            ('What are upgrade rules?', 'No relevant policy was found.'),
        ]:
            with self.subTest(question=question):
                cache = {}
                def store(cache_key, payload, *, ttl_seconds=ANSWER_CACHE_TTL_SECONDS):
                    self.assertEqual(ttl_seconds, 300)
                    cache[cache_key] = payload
                agent = factory.AeroNovaAgent(PromptVersion('test', 'v4', 'prompt', 'hash'), [], object())
                decision = JevRoutingDecision('upgrades', 'public_information', False, False)
                with patch.object(factory, 'create_agent', return_value=agent), \
                     patch.object(factory, 'setup_tracing'), \
                     patch.object(factory, 'get_cached_answer', side_effect=cache.get), \
                     patch.object(factory, 'set_cached_answer', side_effect=store) as write, \
                     patch.object(factory, 'classify_and_route', return_value=decision) as classify, \
                     patch.object(factory, 'hybrid_retrieve', return_value=[]) as retrieve, \
                     patch.object(factory, 'generate_answer') as generate, \
                     patch.object(factory, 'SemanticAnswerCache') as semantic, \
                     patch('builtins.print') as output, \
                     patch('builtins.input', side_effect=[question, question, 'exit']):
                    semantic.return_value.lookup.return_value = None
                    factory.main()
                    classify.assert_called_once_with(question)
                    retrieve.assert_called_once()
                    generate.assert_not_called()
                    write.assert_called_once()
                    semantic.return_value.store.assert_not_called()
                    self.assertEqual(list(cache.values())[0]['cache_kind'], 'no_context')
                    self.assertEqual(list(cache.values())[0]['message'], message)
                    self.assertEqual(sum(call.args == (f'\nAeroNova answer: {message}',) for call in output.call_args_list), 2)
