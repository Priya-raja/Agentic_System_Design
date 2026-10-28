import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from agent import factory
from agent.prompt_manager import PromptVersion
from context.ingest.prepare_documents import split_documents
from context.indexers import retrieve


class DepartmentRoutingTests(unittest.TestCase):
    def test_entire_source_keeps_department_after_chunking(self):
        for source, department in [
            ("POL-CLAIM-2026", "refunds_claims"),
            ("POL-DIS-2026", "disruptions"),
            ("POL-BAG-2026", "baggage"),
            ("CAT-CABIN-MENU-2026", "cabin_onboard"),
            ("WEB-003", "refunds_claims"),
            ("WEB-005", "disruptions"),
            ("OTHER", "general_policies"),
        ]:
            chunks = split_documents(
                [Document(page_content="Checked baggage. " * 200, metadata={"document_id": source})]
            )
            self.assertGreater(len(chunks), 1)
            self.assertTrue(all(chunk.metadata["department"] == department for chunk in chunks))

    def test_department_filter_in_qdrant(self):
        filters = retrieve.create_qdrant_filter(2026, None, "disruptions")
        self.assertEqual(
            {item.key: item.match.value for item in filters.must},
            {"metadata.policy_year": 2026, "metadata.department": "disruptions"},
        )

    def test_old_index_requires_rebuild(self):
        with tempfile.TemporaryDirectory() as folder:
            state = Path(folder) / "index_state.json"
            state.write_text(json.dumps({"collection_name": retrieve.COLLECTION_NAME, "documents": {}}))
            with patch.object(retrieve, "INDEX_STATE_FILE", state):
                with self.assertRaisesRegex(RuntimeError, "department metadata"):
                    retrieve.ensure_index_is_fresh([])

    def test_private_request_cannot_be_answered_from_public_policy(self):
        from agent.session import ChatSession
        from agent.tools.base import ToolRegistry

        fake = factory.AeroNovaAgent(PromptVersion("test", "v4", "prompt", "hash"), [], object())
        synth = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        session = ChatSession(
            agent=fake,
            registry=ToolRegistry(),
            semantic_cache=object(),
            cache_get=lambda _: None,
            synthesizer=synth,
        )
        result = session.ask("Where is my refund?")
        self.assertEqual(result["kind"], "handoff")
        self.assertIn("official website", result["answer"])
        synth.assert_not_called()

    def test_ticket_refund_requests_get_contact_guidance_without_retrieval(self):
        from agent.session import ChatSession
        from agent.tools.base import ToolRegistry

        fake = factory.AeroNovaAgent(PromptVersion("test", "v4", "prompt", "hash"), [], object())
        synth = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        for question in [
            "Refund ticket",
            "Can I get a refund for my cancelled flight?",
            "What is the ticket refund policy?",
            "Is there a refund policy?",
        ]:
            with self.subTest(question=question):
                session = ChatSession(
                    agent=fake,
                    registry=ToolRegistry(),
                    semantic_cache=object(),
                    cache_get=lambda _: None,
                    synthesizer=synth,
                )
                result = session.ask(question)
                self.assertEqual(result["kind"], "handoff")
                self.assertIn("booking reference", result["answer"])
                self.assertIn("Do not email payment-card details", result["answer"])
        synth.assert_not_called()

    def test_baggage_and_eu261_refunds_stay_in_policy_rag(self):
        from agent.answers import is_ticket_refund_handoff

        self.assertFalse(is_ticket_refund_handoff("Can I get a refund for delayed baggage?"))
        self.assertFalse(is_ticket_refund_handoff("What does EU261 say about refunds after cancellation?"))

    def test_refund_email_request_returns_a_draft_without_sending(self):
        from agent.session import ChatSession
        from agent.tools.base import ToolRegistry

        fake = factory.AeroNovaAgent(PromptVersion("test", "v4", "prompt", "hash"), [], object())
        session = ChatSession(
            agent=fake,
            registry=ToolRegistry(),
            semantic_cache=object(),
            cache_get=lambda _: None,
        )
        result = session.ask("Write an email to AeroNova about my cancelled ticket refund")
        self.assertEqual(result["kind"], "handoff")
        self.assertIn("Subject: Ticket refund request", result["answer"])
        self.assertIn("[booking reference]", result["answer"])
        self.assertIn("I cannot send it for you", result["answer"])
