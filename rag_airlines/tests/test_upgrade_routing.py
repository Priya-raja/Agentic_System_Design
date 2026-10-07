import unittest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from agent.jev_classifier import QUESTIONS, classify_and_route, parse_routing_response
from agent.router import route_support
from context.ingest.prepare_documents import split_documents, UPGRADE_DOCUMENT_IDS
from context.indexers.retrieve import hybrid_retrieve


class UpgradeRoutingTests(unittest.TestCase):
    def test_expected_classifications_select_correct_routes(self):
        cases = [
            ('Can I upgrade Economy Flex directly to Business?', 'public_information', 'public_rag'),
            ('Can I upgrade my Economy Flex ticket directly to Business?', 'public_information', 'public_rag'),
            ('What tickets can upgrade to Business?', 'public_information', 'public_rag'),
            ('Is booking AN7K2P eligible for an upgrade?', 'customer_specific', 'upgrade_workflow'),
            ('How much is my upgrade offer?', 'customer_specific', 'upgrade_workflow'),
            ('Is Business upgrade inventory available tomorrow?', 'customer_specific', 'upgrade_workflow'),
        ]
        # Test handling of expected provider responses, not actual model accuracy.
        for question, scope, destination in cases:
            provider = MagicMock()
            provider.json.return_value = {'answers': {
                'department': {'choice': 'upgrades', 'confidence': .99},
                'request_scope': {'choice': scope, 'confidence': .99},
                'model_tier': {'choice': 'cheap', 'confidence': .99},
        'risk_category': {'choice': 'safe', 'confidence': .99},
        'pii_presence': {'choice': 'none', 'confidence': .99},
            }}
            with patch.dict('os.environ', {'OPENCODE_API_KEY': 'test'}), patch('agent.jev_classifier.httpx.post', return_value=provider) as post:
                decision = classify_and_route(question)
                route = route_support(decision)
                self.assertEqual(route.destination, 'secure_service' if 'AN7K2P' in question else destination)
                self.assertEqual(route.requires_authentication, scope == 'customer_specific')
                self.assertFalse(route.needs_human_review)
                self.assertEqual(route.retrieval_topic, 'upgrades' if scope == 'public_information' else None)
                self.assertEqual(post.call_args.kwargs['json']['state'], question.replace('AN7K2P', '[REDACTED_BOOKING_REFERENCE]'))
                self.assertEqual(post.call_args.kwargs['json']['questions'], QUESTIONS)

    def test_upgrade_chunks_keep_topic_and_department(self):
        for source in UPGRADE_DOCUMENT_IDS:
            chunks = split_documents([Document(page_content='Business seat upgrade rules. ' * 100,
                                               metadata={'document_id': source})])
            self.assertTrue(all(c.metadata['department'] == c.metadata['topic'] == 'upgrades' for c in chunks))

    def test_retrieval_excludes_other_departments(self):
        chunks = [Document(page_content='Economy Flex cannot upgrade directly to Business.',
                           metadata={'document_id': source, 'department': 'upgrades', 'topic': 'upgrades',
                                     'chunk_id': source, 'authority': 'primary'})
                  for source in sorted(UPGRADE_DOCUMENT_IDS)]
        other = Document(page_content='Business baggage allowance.', metadata={
            'document_id': 'POL-BAG-2026', 'department': 'baggage', 'topic': 'baggage',
            'chunk_id': 'other', 'authority': 'primary'})
        store = MagicMock()
        # Defense also filters unexpected dense results.
        store.similarity_search_with_score.return_value = [(other, .99), (chunks[0], .8)]
        results = hybrid_retrieve('Can I upgrade Economy Flex directly to Business?',
                                  chunks + [other], store, department='upgrades')
        self.assertEqual({document.metadata['document_id'] for document, _ in results}, UPGRADE_DOCUMENT_IDS)
        filters = store.similarity_search_with_score.call_args.kwargs['filter']
        self.assertEqual(filters.must[0].key, 'metadata.department')
        self.assertEqual(filters.must[0].match.value, 'upgrades')
