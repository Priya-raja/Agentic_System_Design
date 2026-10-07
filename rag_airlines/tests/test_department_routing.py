import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from agent import factory
from agent.jev_classifier import parse_routing_response, fallback_decision
from agent.prompt_manager import PromptVersion
from context.ingest.prepare_documents import split_documents
from context.indexers import retrieve


def response(department, scope, confidence=.99):
    return {'answers': {
        'department': {'choice': department, 'confidence': confidence},
        'request_scope': {'choice': scope, 'confidence': .99},
        'model_tier': {'choice': 'cheap', 'confidence': .99},
        'risk_category': {'choice': 'safe', 'confidence': .99},
        'pii_presence': {'choice': 'none', 'confidence': .99},
    }}


class DepartmentRoutingTests(unittest.TestCase):
    def test_public_refund_rules_vs_customer_refund_status(self):
        public = parse_routing_response(response('refunds_claims', 'public_information'))
        private = parse_routing_response(response('refunds_claims', 'customer_specific'))
        self.assertTrue(public.rag_allowed)
        self.assertFalse(public.needs_authentication)
        self.assertFalse(private.rag_allowed)
        self.assertTrue(private.needs_authentication)

    def test_general_policies_and_support_stay_separate(self):
        self.assertTrue(parse_routing_response(response('general_policies', 'public_information')).rag_allowed)
        support = parse_routing_response(response('general_support', 'public_information'))
        self.assertFalse(support.rag_allowed)
        self.assertTrue(support.needs_human_review)

    def test_inactive_workflows_and_low_confidence_block_rag(self):
        for department in ['rewards_loyalty', 'booking_changes']:
            self.assertFalse(parse_routing_response(response(department, 'public_information')).rag_allowed)
        low = parse_routing_response(response('baggage', 'public_information', .1))
        self.assertEqual(low.department, 'general_support')
        self.assertFalse(low.rag_allowed)
        scoped = response('baggage', 'public_information')
        scoped['answers']['request_scope']['confidence'] = .1
        self.assertFalse(parse_routing_response(scoped).rag_allowed)
        self.assertFalse(fallback_decision('test missing credentials').rag_allowed)

    def test_public_upgrade_scope_068_is_accepted(self):
        data = response('upgrades', 'public_information')
        data['answers']['request_scope']['confidence'] = .68
        decision = parse_routing_response(data)
        self.assertEqual(decision.department, 'upgrades')
        self.assertTrue(decision.rag_allowed)
        self.assertFalse(decision.needs_authentication)

    def test_uncertain_scope_preserves_known_department_and_requires_review(self):
        data = response('upgrades', 'public_information')
        data['answers']['request_scope']['confidence'] = .1
        decision = parse_routing_response(data)
        self.assertEqual(decision.department, 'upgrades')
        self.assertTrue(decision.needs_human_review)
        self.assertFalse(decision.rag_allowed)

    def test_invalid_choices_and_confidences_rejected(self):
        for department, confidence in [('other', .99), ('baggage', float('nan')), ('baggage', 2)]:
            with self.assertRaises(ValueError):
                parse_routing_response(response(department, 'public_information', confidence))

    def test_entire_source_keeps_department_after_chunking(self):
        for source, department in [('POL-CLAIM-2026', 'refunds_claims'),
                                   ('POL-DIS-2026', 'disruptions'),
                                   ('POL-BAG-2026', 'baggage'),
                                   ('CAT-CABIN-MENU-2026', 'cabin_onboard'),
                                   ('WEB-003', 'refunds_claims'),
                                   ('WEB-005', 'disruptions'),
                                   ('OTHER', 'general_policies')]:
            chunks = split_documents([Document(page_content='Checked baggage. ' * 200,
                                               metadata={'document_id': source})])
            self.assertGreater(len(chunks), 1)
            self.assertTrue(all(chunk.metadata['department'] == department for chunk in chunks))

    def test_department_filter_in_qdrant(self):
        filters = retrieve.create_qdrant_filter(2026, None, 'disruptions')
        self.assertEqual({item.key: item.match.value for item in filters.must},
                         {'metadata.policy_year': 2026, 'metadata.department': 'disruptions'})

    def test_old_index_requires_rebuild(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / 'index_state.json'
            state.write_text(json.dumps({'collection_name': retrieve.COLLECTION_NAME, 'documents': {}}))
            with patch.object(retrieve, 'INDEX_STATE_FILE', state):
                with self.assertRaisesRegex(RuntimeError, 'department metadata'):
                    retrieve.ensure_index_is_fresh([])

    def test_private_cli_request_skips_retrieval_generation_and_semantic_cache(self):
        agent = factory.AeroNovaAgent(PromptVersion('test', 'v4', 'prompt', 'hash'), [], object())
        decision = parse_routing_response(response('refunds_claims', 'customer_specific'))
        with patch.object(factory, 'create_agent', return_value=agent), \
             patch.object(factory, 'setup_tracing'), \
             patch.object(factory, 'get_cached_answer', return_value=None), \
             patch.object(factory, 'classify_and_route', return_value=decision), \
             patch.object(factory, 'hybrid_retrieve') as retrieve_mock, \
             patch.object(factory, 'generate_answer') as generate, \
             patch.object(factory, 'SemanticAnswerCache') as semantic, \
             patch('builtins.input', side_effect=['Where is my refund?', 'exit']):
            factory.main()
            retrieve_mock.assert_not_called()
            generate.assert_not_called()
            semantic.return_value.lookup.assert_not_called()
            semantic.return_value.store.assert_not_called()


if __name__ == '__main__':
    unittest.main()
