import json
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import ROOT_DIR
from agent.refund_agent import RefundAgent, RefundError
from context.ingest.prepare_documents import prepare_public_chunks
from contextlib import redirect_stdout
from io import StringIO


class RefundAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.crm = self.root / "crm"
        self.crm.mkdir()
        for name in ["refund_claims.jsonl", "baggage_cases.jsonl", "bookings.jsonl"]:
            shutil.copy(ROOT_DIR / "data/crm" / name, self.crm / name)
        self.agent = RefundAgent(
            demo=True, customer_id="CUST-1001", crm_dir=self.crm, ledger=self.root / "ledger.sqlite3"
        )

    def mutate(self, **fields):
        path = self.crm / "refund_claims.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0].update(fields)
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    def test_review_and_confirmation(self):
        self.assertTrue(self.agent.check("REF-DEMO-001")["can_proceed"])
        with self.assertRaises(RefundError):
            self.agent.proceed("REF-DEMO-001")
        self.assertFalse(self.agent.ledger.exists())

    def test_simulation_idempotent_and_check_updates(self):
        first = self.agent.proceed("REF-DEMO-001", confirmed=True)
        second = self.agent.proceed("REF-DEMO-001", confirmed=True)
        self.assertFalse(first["already_processed"])
        self.assertTrue(second["already_processed"])
        self.assertEqual(first["receipt_id"], second["receipt_id"])
        self.assertFalse(self.agent.check("REF-DEMO-001")["can_proceed"])

    def test_concurrent_proceed_records_once(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.agent.proceed("REF-DEMO-001", confirmed=True), range(2)))
        self.assertEqual(sum(not row["already_processed"] for row in results), 1)

    def test_other_customer_and_missing_are_indistinguishable(self):
        for claim in ["REF-DEMO-002", "DOES-NOT-EXIST"]:
            with self.assertRaisesRegex(RefundError, "Claim not found for this demo customer"):
                self.agent.check(claim)
        self.assertEqual(len(self.agent.list_claims()), 1)

    def test_pending_rejected_invalid_amount_and_destination_fail(self):
        for fields in [
            {"status": "pending_review"},
            {"status": "rejected"},
            {"approved_amount_aed": "10000"},
            {"approved_amount_aed": "NaN"},
            {"approved_amount_aed": "-1"},
            {"evidence_verified": False},
            {"approval_reference": None},
            {"payment_destination": "other_account"},
        ]:
            with self.subTest(fields=fields):
                original = (ROOT_DIR / "data/crm/refund_claims.jsonl").read_text()
                (self.crm / "refund_claims.jsonl").write_text(original)
                self.mutate(**fields)
                with self.assertRaises(RefundError):
                    self.agent.proceed("REF-DEMO-001", confirmed=True)

    def test_changed_processed_claim_requires_review(self):
        self.agent.proceed("REF-DEMO-001", confirmed=True)
        self.mutate(approved_amount_aed="100.00")
        with self.assertRaisesRegex(RefundError, "has changed"):
            self.agent.proceed("REF-DEMO-001", confirmed=True)

    def test_chat_commands_use_bound_demo_agent_without_llm_or_cache(self):
        from agent.cli import refund_command

        with redirect_stdout(StringIO()):
            refund_command("/refund proceed REF-DEMO-001 confirm", self.agent)
        self.assertEqual(self.agent.check("REF-DEMO-001")["refund_status"], "simulated_processed")

    def test_real_data_and_unbound_mode_rejected(self):
        with self.assertRaises(RefundError):
            RefundAgent(demo=False, customer_id="CUST-1001")
        self.mutate(synthetic=False)
        with self.assertRaises(RefundError):
            self.agent.check("REF-DEMO-001")


class RewardsCorpusTests(unittest.TestCase):
    def test_rewards_documents_public_and_crm_excluded(self):
        with redirect_stdout(StringIO()):
            chunks = prepare_public_chunks()
        rewards = [chunk for chunk in chunks if chunk.metadata["department"] == "rewards_loyalty"]
        self.assertEqual(
            {chunk.metadata["document_id"] for chunk in rewards},
            {"POL-REWARDS-2026", "POL-REDEEM-2026", "WEB-008"},
        )
        self.assertTrue(all(chunk.metadata["topic"] == "rewards_loyalty" for chunk in rewards))
        self.assertTrue(any(chunk.metadata["authority"] == "primary" for chunk in rewards))
        self.assertTrue(all(chunk.metadata["corpus_folder"] in {"policies", "web"} for chunk in chunks))
