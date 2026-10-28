from unittest.mock import MagicMock
import pytest
from langchain_core.documents import Document
from config import ROOT_DIR
from context.routes import menu_route
from context.ingest.load_documents import load_pdf
from context.ingest.prepare_documents import select_public_documents, split_documents
from context.indexers.retrieve import hybrid_retrieve


@pytest.mark.parametrize(
    "question, expected",
    [
        ("Menu from DXB to London", "DXB-LHR"),
        ("Menu from London to DXB", "LHR-DXB"),
        ("Meals from Dubai to Mumbai", "DXB-BOM"),
        ("Drinks from Singapore to Dubai", "SIN-DXB"),
        ("Baggage from Dubai to London", None),
        ("Compare menus from DXB to LHR and LHR to DXB", None),
    ],
)
def test_directional_aliases(question, expected):
    assert menu_route(question) == expected


def test_pdf_pages_keep_route_scope_after_chunking():
    documents = load_pdf(ROOT_DIR / "data/policies/CAT-CABIN-ROUTES-2026.pdf")
    assert len(documents) == 6
    chunks = split_documents(select_public_documents(documents))
    expected = {1: "DXB-LHR", 2: "LHR-DXB", 3: "DXB-BOM", 4: "BOM-DXB", 5: "DXB-SIN", 6: "SIN-DXB"}
    for chunk in chunks:
        assert chunk.metadata["department"] == "cabin_onboard"
        assert chunk.metadata["route"] == expected[chunk.metadata["page"]]
        assert chunk.metadata["authority"] == "primary"
    assert {c.metadata["route"] for c in chunks} == set(expected.values())


def test_both_searches_and_final_results_exclude_reverse_route():
    chunks = [
        Document(
            page_content=text,
            metadata={
                "chunk_id": route,
                "route": route,
                "department": "cabin_onboard",
                "topic": "menu",
                "authority": "primary",
                "policy_year": 2026,
            },
        )
        for route, text in [("DXB-LHR", "Outbound menu with chicken"), ("LHR-DXB", "Return menu with pasta")]
    ]
    vector = MagicMock()
    # Deliberately misbehaving vector mock: defensive filter must still reject outbound.
    vector.similarity_search_with_score.return_value = [(doc, 0.9) for doc in chunks]
    result = hybrid_retrieve("Menu from London to Dubai", chunks, vector, department="cabin_onboard")
    assert result and all(doc.metadata["route"] == "LHR-DXB" for doc, _ in result)
    conditions = vector.similarity_search_with_score.call_args.kwargs["filter"].must
    assert any(c.key == "metadata.route" and c.match.value == "LHR-DXB" for c in conditions)


def test_dense_search_failure_falls_back_to_department_filtered_bm25():
    chunk = Document(
        page_content="The London to Dubai menu includes a vegetarian dinner and refreshments.",
        metadata={
            "chunk_id": "MENU-LHR-DXB",
            "route": "LHR-DXB",
            "department": "cabin_onboard",
            "topic": "menu",
            "authority": "primary",
            "policy_year": 2026,
        },
    )
    vector = MagicMock()
    vector.similarity_search_with_score.side_effect = ConnectionError("embedding service unavailable")

    result = hybrid_retrieve("Menu from London to Dubai", [chunk], vector, department="cabin_onboard")

    assert result
    assert result[0][0].metadata["chunk_id"] == "MENU-LHR-DXB"
