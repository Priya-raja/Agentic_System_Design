import json
import unittest
from unittest.mock import patch

from agent import factory
from agent.prompt_manager import PromptVersion
from config import ANSWER_CACHE_TTL_SECONDS, NO_CONTEXT_CACHE_TTL_SECONDS
from context.memory import answer_cache


class NoContextCacheTests(unittest.TestCase):
    def test_redis_expiry_override_and_default(self):
        with patch.object(answer_cache, "redis_client") as redis:
            answer_cache.set_cached_answer("no-context", {"message": "missing"}, ttl_seconds=300)
            self.assertEqual(redis.set.call_args.kwargs["ex"], NO_CONTEXT_CACHE_TTL_SECONDS)
            self.assertEqual(json.loads(redis.set.call_args.args[1]), {"message": "missing"})
            answer_cache.set_cached_answer("answer", {"response": {}})
            self.assertEqual(redis.set.call_args.kwargs["ex"], ANSWER_CACHE_TTL_SECONDS)

    def test_repeated_no_context_skips_tools_and_generation(self):
        from unittest.mock import MagicMock
        from agent.session import ChatSession
        from agent.tools.base import ToolRegistry, Tool
        from agent.tools.policy_tool import PolicyArgs

        for year in [2025, None]:
            question = "What are upgrade rules?" + (" in 2025" if year else "")
            cache = {}

            def write(key, payload, **kwargs):
                self.assertEqual(kwargs.get("ttl_seconds"), 300)
                cache[key] = payload

            handler = MagicMock(return_value={"department": "upgrades", "chunks": []})
            registry = ToolRegistry().register(Tool("retrieve_policy", PolicyArgs, handler))
            semantic = MagicMock()
            semantic.lookup.return_value = None
            synth = MagicMock()
            agent = factory.AeroNovaAgent(PromptVersion("test", "v4", "prompt", "hash"), [], object())
            session = ChatSession(
                agent,
                registry,
                semantic,
                synthesizer=synth,
                cache_get=cache.get,
                cache_set=write,
            )
            first = session.ask(question)
            second = session.ask(question)
            self.assertEqual(first["kind"], "no_context")
            self.assertEqual(second["cache"], "exact")
            handler.assert_called_once()
            synth.assert_not_called()
