"""Select public RAG or a secure workflow from a classified request."""
from dataclasses import dataclass
from agent.jev_classifier import JevRoutingDecision
from safety.policy import decide_safety_action
from safety.heuristics import PIIResult, BLOCK, REVIEW, SECURE


@dataclass(frozen=True)
class SupportRoute:
    department: str
    destination: str
    retrieval_topic: str | None
    model_tier: str
    requires_authentication: bool
    needs_human_review: bool
    reason: str


def route_support(decision: JevRoutingDecision) -> SupportRoute:
    safety = decide_safety_action(local_pii=PIIResult(False, (), ""),
                                  jev_pii_probability=decision.pii_probability,
                                  harmful_probability=decision.harmful_probability,
                                  risk_category=decision.risk_category,
                                  risk_confidence=decision.risk_confidence)
    if safety.action in {BLOCK, REVIEW}:
        return SupportRoute(decision.department, "safety_block" if safety.action == BLOCK else "human_support",
                            None, "standard", False, True, safety.reason)
    if safety.action == SECURE:
        return SupportRoute(decision.department, "secure_service", None, "standard", True, False, safety.reason)
    if decision.needs_human_review:
        return SupportRoute(decision.department, "human_support", None, "standard",
                            decision.needs_authentication, True,
                            "Unsupported department or uncertain classification.")
    if decision.request_scope == "customer_specific":
        workflows = {
            "upgrades": "upgrade_workflow", "refunds_claims": "claims_workflow",
            "rewards_loyalty": "rewards_workflow", "booking_changes": "booking_workflow",
            "disruptions": "operations_workflow",
        }
        return SupportRoute(decision.department, workflows.get(decision.department, "customer_service_workflow"),
                            None, "standard", True, False,
                            "The request requires customer records or live inventory data.")
    if decision.rag_allowed:
        return SupportRoute(decision.department, "public_rag", decision.department,
                            decision.model_tier, False, False, "Public policy question.")
    return SupportRoute(decision.department, "human_support", None, "standard",
                        decision.needs_authentication, True, "No supported public RAG route.")
