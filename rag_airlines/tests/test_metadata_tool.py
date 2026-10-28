from langchain_core.documents import Document

from agent.cascade import classify_public_department, route_public_intent
from agent.tools.metadata_tool import MetadataTool
from agent.verifier import Verifier
from agent.answers import GroundedAnswer


def test_eu_law_metadata_question_uses_metadata_only_tool():
    question = "Which parts of this answer come from EU law, and which are AeroNova policy?"
    assert route_public_intent(question) == "inspect_metadata"


def test_extraordinary_circumstances_routes_to_disruptions():
    assert classify_public_department("When might extraordinary circumstances affect compensation?") == "disruptions"


def test_metadata_tool_returns_source_facts_and_verifiable_citations():
    chunks = [
        Document(page_content="not exposed", metadata={
            "document_id": "POL-EU261-2004", "source_file": "POL-EU261-2004.md",
            "authority": "primary", "department": "disruptions", "status": "active",
            "valid_from": "2005-02-17", "valid_to": "2027-10-22",
        }),
        Document(page_content="also not exposed", metadata={
            "document_id": "POL-DIS-2026", "source_file": "POL-DIS-2026.pdf",
            "authority": "primary", "department": "disruptions", "status": "active",
            "valid_from": "2026-01-01", "valid_to": "2026-12-31",
        }),
    ]
    output = MetadataTool(chunks).run(query="Which documents are EU law or AeroNova policy?")
    assert output.ok
    assert all("not exposed" not in str(item) for item in output.data["documents"])
    eu_document = next(item for item in output.data["documents"] if item["document_id"] == "POL-EU261-2004")
    assert eu_document["source_class"] == "EU law"
    verifier = Verifier(tool_results=[output])
    assert {doc.metadata["document_id"] for doc, _ in verifier.results} == {"POL-EU261-2004", "POL-DIS-2026"}
    assert verifier.check(GroundedAnswer(
        answerable=True,
        answer="POL-EU261-2004 is EU law; POL-DIS-2026 is AeroNova demo policy.",
        citation_chunk_ids=["META-POL-EU261-2004", "META-POL-DIS-2026"],
    )).passed
