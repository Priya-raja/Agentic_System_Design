"""CLI presentation only. No retrieval, safety or synthesis decisions."""

import json
from agent.refund_agent import RefundError
from agent.refund_graph import RefundWorkflow


def print_result(result):
    print(f"\nAERONOVA [{result['cache'].upper()}] - {result['kind']} ({result['latency_ms']:.0f} ms)")
    print(result["answer"])
    for source in result.get("sources", []):
        print(f"- {source['document_id']} | p.{source['page']} | {source['authority']}")
    if "private_data" in result:
        print(json.dumps(result["private_data"], indent=2, default=str))


def refund_command(question, refund_agent):
    if refund_agent is None:
        print("Use --demo-refund-customer to bind a synthetic demo customer. This is not authentication.")
        return
    parts = question.split()
    try:
        if parts == ["/refund", "list"]:
            result = refund_agent.list_claims()
        elif len(parts) == 3 and parts[1] == "check":
            result = refund_agent.check(parts[2])
        elif len(parts) == 3 and parts[1] == "review":
            result = RefundWorkflow(refund_agent).start(parts[2])
        elif len(parts) == 4 and parts[1] == "approve" and parts[3] == "confirm":
            result = RefundWorkflow(refund_agent).decide(parts[2], True)
        elif len(parts) == 3 and parts[1] == "reject":
            result = RefundWorkflow(refund_agent).decide(parts[2], False)
        elif len(parts) == 4 and parts[1] == "proceed" and parts[3] == "confirm":
            flow = RefundWorkflow(refund_agent)
            review = flow.start(parts[2])
            result = review if review["status"] == "completed" else flow.decide(review["workflow_id"], True)
        else:
            print("Use /refund list, /refund check CLAIM_ID, /refund review CLAIM_ID, /refund approve REVIEW_ID confirm, or /refund reject REVIEW_ID. Demo only.")
            return
        print(json.dumps(result, indent=2))
    except RefundError as error:
        print(f"Refund not processed: {error}")
