from langchain_core.documents import Document
from agent.answers import GroundedAnswer
from agent.verifier import Verifier
from agent.tools.base import ToolResult


def evidence(authority="primary", department="baggage", text="Allowance is 25 kg."):
    return [
        (
            Document(
                page_content=text,
                metadata={"chunk_id": "C1", "authority": authority, "department": department},
            ),
            1,
        )
    ]


def test_fabricated_citations_and_measurements_rejected():
    verifier = Verifier(evidence())
    assert not verifier.check(
        GroundedAnswer(answerable=True, answer="25 kg", citation_chunk_ids=["fake"])
    ).passed
    assert not verifier.check(
        GroundedAnswer(answerable=True, answer="40 kg", citation_chunk_ids=["C1"])
    ).passed
    assert verifier.check(GroundedAnswer(answerable=True, answer="25 kg", citation_chunk_ids=["C1"])).passed


def test_unsafe_outputs_and_authority_policy():
    response = GroundedAnswer(answerable=True, answer="Contact person@example.com", citation_chunk_ids=["C1"])
    assert not Verifier(evidence()).check(response).passed
    assert Verifier(evidence("secondary")).evidence_errors()
    assert not Verifier(evidence("secondary", "cabin_onboard")).evidence_errors()


def test_tool_facts_have_their_own_verifiable_source_and_failed_tools_are_not_evidence():
    tool = ToolResult(tool="flight_status", ok=True, data={"delay": "45 minutes", "live": False}, call_id=2)
    verifier = Verifier(tool_results=[tool])
    assert verifier.check(
        GroundedAnswer(answerable=True, answer="45 minutes", citation_chunk_ids=["TOOL-2"])
    ).passed
    assert not Verifier(tool_results=[ToolResult(tool="flight_status", ok=False)]).results


def test_regulatory_time_quantities_accept_equivalent_units_but_not_changed_values():
    source = evidence(department="disruptions", text="Passengers may wait 24 hours.")
    assert Verifier(source).check(
        GroundedAnswer(answerable=True, answer="Passengers may wait one day.", citation_chunk_ids=["C1"])
    ).passed
    assert not Verifier(source).check(
        GroundedAnswer(answerable=True, answer="Passengers may wait 48 hours.", citation_chunk_ids=["C1"])
    ).passed
