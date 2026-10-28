import unittest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from agent.cascade import classify_public_department, route_public_intent, needs_customer_workflow
from context.ingest.prepare_documents import split_documents, UPGRADE_DOCUMENT_IDS
from context.indexers.retrieve import hybrid_retrieve


class UpgradeRoutingTests(unittest.TestCase):
    def test_deterministic_intent_routes_upgrade_questions(self):
        for question in [
            "Can I upgrade Economy Flex directly to Business?",
            "Can I upgrade my Economy Flex ticket directly to Business?",
            "What tickets can upgrade to Business?",
            "Is booking AN7K2P eligible for an upgrade?",
        ]:
            with self.subTest(question=question):
                self.assertEqual(classify_public_department(question), "upgrades")
                self.assertEqual(route_public_intent(question), "upgrades")

    def test_customer_specific_offer_and_inventory_are_handed_off(self):
        for question in [
            "How much is my upgrade offer?",
            "Is Business upgrade inventory available tomorrow?",
        ]:
            with self.subTest(question=question):
                self.assertTrue(needs_customer_workflow(question))

    def test_upgrade_chunks_keep_topic_and_department(self):
        for source in UPGRADE_DOCUMENT_IDS:
            chunks = split_documents(
                [
                    Document(
                        page_content="Business seat upgrade rules. " * 100,
                        metadata={"document_id": source},
                    )
                ]
            )
            self.assertTrue(
                all(c.metadata["department"] == c.metadata["topic"] == "upgrades" for c in chunks)
            )

    def test_retrieval_excludes_other_departments(self):
        chunks = [
            Document(
                page_content="Economy Flex cannot upgrade directly to Business.",
                metadata={
                    "document_id": source,
                    "department": "upgrades",
                    "topic": "upgrades",
                    "chunk_id": source,
                    "authority": "primary",
                },
            )
            for source in sorted(UPGRADE_DOCUMENT_IDS)
        ]
        other = Document(
            page_content="Business baggage allowance.",
            metadata={
                "document_id": "POL-BAG-2026",
                "department": "baggage",
                "topic": "baggage",
                "chunk_id": "other",
                "authority": "primary",
            },
        )
        store = MagicMock()
        store.similarity_search_with_score.return_value = [(other, 0.99), (chunks[0], 0.8)]
        results = hybrid_retrieve(
            "Can I upgrade Economy Flex directly to Business?",
            chunks + [other],
            store,
            department="upgrades",
        )
        self.assertEqual({document.metadata["document_id"] for document, _ in results}, UPGRADE_DOCUMENT_IDS)
        filters = store.similarity_search_with_score.call_args.kwargs["filter"]
        self.assertEqual(filters.must[0].key, "metadata.department")
        self.assertEqual(filters.must[0].match.value, "upgrades")
