"""Durable, customer-bound LangGraph approval for simulated refunds only."""

import sqlite3
from contextlib import closing
import threading
import uuid
from pathlib import Path
from typing import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from agent.refund_agent import RefundError


class RefundState(TypedDict, total=False):
    customer_id: str
    claim_id: str
    fingerprint: str
    review: dict
    approved: bool
    result: dict


def build_refund_graph(refunds, checkpointer):
    def prepare(state):
        review = refunds.check(state["claim_id"])
        if review["refund_status"] == "simulated_processed":
            return {"result": {**review, "already_processed": True}}
        if not review["can_proceed"]:
            raise RefundError(review["reason"])
        return {"review": review, "fingerprint": refunds.review_fingerprint(state["claim_id"])}

    def human_review(state):
        # No side effects before interrupt: this node restarts on resume.
        decision = interrupt(
            {
                "action": "refund_simulation",
                "claim_id": state["claim_id"],
                "amount_aed": state["review"]["approved_amount_aed"],
                "message": "Approve or reject this simulated refund. No payment will be sent.",
            }
        )
        if not isinstance(decision, dict) or type(decision.get("approved")) is not bool:
            raise RefundError("A human approval or rejection is required.")
        return {"approved": decision["approved"]}

    def process(state):
        if state.get("approved") is not True:
            raise RefundError("Human approval is required.")
        if state["fingerprint"] != refunds.review_fingerprint(state["claim_id"]):
            raise RefundError("Claim evidence changed after review. Start a new review before approving.")
        return {
            "result": refunds.proceed(
                state["claim_id"], confirmed=True, expected_fingerprint=state["fingerprint"]
            )
        }

    def reject(state):
        return {
            "result": {
                "claim_id": state["claim_id"],
                "refund_status": "rejected",
                "simulation": True,
                "message": "Review rejected. No refund was recorded.",
            }
        }

    graph = StateGraph(RefundState)
    for name, node in [
        ("prepare", prepare),
        ("human_review", human_review),
        ("process", process),
        ("reject", reject),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges(
        "prepare", lambda s: "done" if "result" in s else "review", {"done": END, "review": "human_review"}
    )
    graph.add_conditional_edges(
        "human_review",
        lambda s: "process" if s["approved"] else "reject",
        {"process": "process", "reject": "reject"},
    )
    graph.add_edge("process", END)
    graph.add_edge("reject", END)
    return graph.compile(checkpointer=checkpointer, name="aeronova_refund_approval")


class RefundWorkflow:
    """Each operation opens/closes SQLite; the app serves one local worker.

    Checkpoints contain synthetic review facts only. Never accept a client-provided
    customer identity, graph state, checkpoint ID, or arbitrary Command update.
    """

    def __init__(self, refunds, path: Path | None = None):
        self.refunds = refunds
        self.path = Path(path or refunds.ledger.with_name("approvals.sqlite3"))
        self.lock = threading.RLock()

    def _config(self, workflow_id):
        try:
            if uuid.UUID(workflow_id).hex != workflow_id:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            raise RefundError("Review not found for this demo customer.")
        return {"configurable": {"thread_id": workflow_id}}

    def _graph(self, connection):
        return build_refund_graph(self.refunds, SqliteSaver(connection))

    def _owned(self, graph, config):
        snapshot = graph.get_state(config)
        if not snapshot.values or snapshot.values.get("customer_id") != self.refunds.customer_id:
            raise RefundError("Review not found for this demo customer.")
        # Re-check current record ownership, not just the saved customer ID.
        self.refunds.check(snapshot.values["claim_id"])
        return snapshot

    @staticmethod
    def _view(workflow_id, snapshot):
        values = snapshot.values
        result = values.get("result")
        status = (
            ("rejected" if result.get("refund_status") == "rejected" else "completed")
            if result
            else "pending"
        )
        return {
            "workflow_id": workflow_id,
            "status": status,
            "claim_id": values["claim_id"],
            "review": values.get("review"),
            "result": result,
        }

    def start(self, claim_id):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, closing(sqlite3.connect(self.path, check_same_thread=False)) as connection:
            graph = self._graph(connection)
            workflow_id = uuid.uuid4().hex
            config = self._config(workflow_id)
            # Validate scope before anything is saved.
            self.refunds.check(claim_id)
            graph.invoke({"claim_id": claim_id, "customer_id": self.refunds.customer_id}, config)
            return self._view(workflow_id, graph.get_state(config))

    def get(self, workflow_id):
        config = self._config(workflow_id)
        if not self.path.exists():
            raise RefundError("Review not found for this demo customer.")
        with self.lock, closing(sqlite3.connect(self.path, check_same_thread=False)) as connection:
            graph = self._graph(connection)
            return self._view(workflow_id, self._owned(graph, config))

    def decide(self, workflow_id, approved):
        if type(approved) is not bool:
            raise RefundError("Approval must be true or false.")
        config = self._config(workflow_id)
        if not self.path.exists():
            raise RefundError("Review not found for this demo customer.")
        with self.lock, closing(sqlite3.connect(self.path, check_same_thread=False)) as connection:
            graph = self._graph(connection)
            snapshot = self._owned(graph, config)
            if "result" in snapshot.values:
                view = self._view(workflow_id, snapshot)
                if view["status"] == "completed":
                    view["result"] = {**view["result"], "already_processed": True}
                return view
            if not any(task.interrupts for task in snapshot.tasks):
                raise RefundError("This review is no longer awaiting approval. Start a new review.")
            graph.invoke(Command(resume={"approved": approved}), config)
            return self._view(workflow_id, graph.get_state(config))
