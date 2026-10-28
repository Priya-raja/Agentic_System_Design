import json
import shutil
from unittest.mock import patch
import pytest
from agent.refund_agent import RefundAgent, RefundError
from agent.refund_graph import RefundWorkflow
from config import ROOT_DIR


def make_agent(tmp_path, customer="CUST-1001"):
    return RefundAgent(demo=True, customer_id=customer, ledger=tmp_path / "refunds.sqlite3")


def test_interrupt_survives_restart_and_requires_explicit_decision(tmp_path):
    agent = make_agent(tmp_path)
    flow = RefundWorkflow(agent)
    with patch.object(agent, "proceed", wraps=agent.proceed) as process:
        review = flow.start("REF-DEMO-001")
        assert review["status"] == "pending"
        assert review["review"]["approved_amount_aed"]
        process.assert_not_called()
        assert not agent.ledger.exists()
    resumed = RefundWorkflow(make_agent(tmp_path))
    assert resumed.get(review["workflow_id"])["status"] == "pending"
    result = resumed.decide(review["workflow_id"], True)
    assert result["status"] == "completed"
    assert result["result"]["refund_status"] == "simulated_processed"
    again = resumed.decide(review["workflow_id"], True)
    assert again["result"]["receipt_id"] == result["result"]["receipt_id"]
    assert again["result"]["already_processed"]


def test_rejection_cannot_later_be_promoted_to_approval(tmp_path):
    agent = make_agent(tmp_path)
    flow = RefundWorkflow(agent)
    review = flow.start("REF-DEMO-001")
    assert flow.decide(review["workflow_id"], False)["status"] == "rejected"
    assert flow.decide(review["workflow_id"], True)["status"] == "rejected"
    assert not agent.ledger.exists()


def test_other_customer_invalid_id_and_non_boolean_cannot_resume(tmp_path):
    flow = RefundWorkflow(make_agent(tmp_path))
    review = flow.start("REF-DEMO-001")
    other = RefundWorkflow(make_agent(tmp_path, "CUST-1002"))
    for method in [
        lambda: other.get(review["workflow_id"]),
        lambda: other.decide(review["workflow_id"], True),
        lambda: flow.decide("made-up-id", True),
        lambda: flow.decide(review["workflow_id"], "true"),
    ]:
        with pytest.raises(RefundError):
            method()
    assert flow.get(review["workflow_id"])["status"] == "pending"


def test_changed_receipt_evidence_blocks_approved_action(tmp_path):
    crm = tmp_path / "crm"
    shutil.copytree(ROOT_DIR / "data/crm", crm)
    agent = RefundAgent(demo=True, customer_id="CUST-1001", crm_dir=crm, ledger=tmp_path / "refunds.sqlite3")
    flow = RefundWorkflow(agent)
    review = flow.start("REF-DEMO-001")
    path = crm / "refund_claims.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows[0]["review_note"] = "Changed after approval was requested"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(RefundError, match="changed"):
        flow.decide(review["workflow_id"], True)
    assert not agent.ledger.exists()
