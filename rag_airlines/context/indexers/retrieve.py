from config import ROOT_DIR, INDEX_SCHEMA_VERSION
import json
import os
import re
from datetime import date, datetime

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from opentelemetry import trace
from observability.timing import measure_step

from context.ingest.prepare_documents import prepare_public_chunks
from config import (
    BM25_TOP_K,
    BM25_WEIGHT,
    DENSE_TOP_K,
    DENSE_WEIGHT,
    FINAL_TOP_K,
    PRIMARY_AUTHORITY_BOOST,
    QDRANT_COLLECTION,
    QDRANT_URL,
)

QDRANT_URL = QDRANT_URL
COLLECTION_NAME = QDRANT_COLLECTION
INDEX_STATE_FILE = ROOT_DIR / "index_state.json"

tracer = trace.get_tracer(__name__)

def build_source_state(chunks: list[Document]) -> dict[str, str]:
    return {
        chunk.metadata["source_file"]: chunk.metadata["content_hash"]
        for chunk in chunks
    }


def ensure_index_is_fresh(chunks: list[Document]) -> None:
    """Stop retrieval when Qdrant and local corpus may be out of sync."""

    if not INDEX_STATE_FILE.exists():
        raise RuntimeError(
            "index_state.json is missing. Run build_index.py before "
            "retrieval."
        )

    index_state = json.loads(
        INDEX_STATE_FILE.read_text(encoding="utf-8")
    )

    if index_state.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise RuntimeError(
            "The index needs department metadata. Run python -m "
            "context.indexers.build_index to rebuild it."
        )

    if index_state.get("collection_name") != COLLECTION_NAME:
        raise RuntimeError(
            "The index state belongs to a different Qdrant collection. "
            "Run build_index.py again."
        )

    indexed_documents = index_state.get("documents", {})
    current_documents = build_source_state(chunks)

    if indexed_documents != current_documents:
        changed = sorted(
            source_file
            for source_file in (
                indexed_documents.keys() | current_documents.keys()
            )
            if indexed_documents.get(source_file)
            != current_documents.get(source_file)
        )

        raise RuntimeError(
            "The AeroNova corpus changed after the Qdrant index was "
            "built. Run build_index.py again. Changed sources: "
            + ", ".join(changed)
        )


def extract_policy_year(question: str) -> int | None:
    """Extract a four-digit year such as 2025 from the question."""

    match = re.search(r"\b(20\d{2})\b", question)

    if match:
        return int(match.group())

    return None
def infer_question_topic(question: str) -> str | None:
    """Choose a metadata filter locally; ambiguous questions remain unfiltered."""
    patterns = {
        "menu": r"\b(meals?|menus?|food|drinks?|vegetarian|dietary)\b",
        "baggage": r"\b(baggage|bags?|luggage|power banks?)\b",
        "seating": r"\b(seats?|seating|recline|pitch|lie[- ]flat)\b",
    }
    matches = [topic for topic, pattern in patterns.items()
               if re.search(pattern, question, flags=re.IGNORECASE)]
    return matches[0] if len(matches) == 1 else None


def extract_requested_date(question: str) -> date | None:
    """Extract a month/year or year from a question."""

    month_names = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    lowered = question.lower()
    year = extract_policy_year(question)

    if year is None:
        return None

    for month_name, month_number in month_names.items():
        if month_name in lowered:
            return date(year, month_number, 1)

    return date(year, 1, 1)


def document_is_valid_for_date(
    document: Document,
    requested_date: date | None,
) -> bool:
    """Reject documents outside their declared validity window."""

    if requested_date is None:
        return True

    valid_from = document.metadata.get("valid_from")
    valid_to = document.metadata.get("valid_to")

    if not valid_from or not valid_to:
        return False

    start = datetime.strptime(valid_from, "%Y-%m-%d").date()
    end = datetime.strptime(valid_to, "%Y-%m-%d").date()

    return start <= requested_date <= end


def filter_chunks_by_year(
    chunks: list[Document],
    year: int | None,
) -> list[Document]:
    """
    Restrict BM25 documents to the requested policy year.

    If the question does not contain a year, return every chunk.
    """

    if year is None:
        return chunks

    return [
        chunk
        for chunk in chunks
        if chunk.metadata.get("policy_year") == year
    ]


def create_qdrant_filter(
    year: int | None,
    topic: str | None,
    department: str | None = None,
) -> Filter | None:

    """Create the equivalent year and topic filter for Qdrant."""

    conditions = []

    if year is not None:
        conditions.append(
            FieldCondition(
                key="metadata.policy_year",
                match=MatchValue(value=year),
            )
        )

    if topic is not None:
        conditions.append(
            FieldCondition(
                key="metadata.topic",
                match=MatchValue(value=topic),
            )
        )

    if department is not None:
        conditions.append(FieldCondition(
            key="metadata.department", match=MatchValue(value=department),
        ))

    if not conditions:
        return None

    return Filter(must=conditions)


# RRF Function for Merging Results from Multiple Retrievers
def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    weights: list[float],
    rank_constant: int = 60,
) -> list[tuple[Document, float]]:
    """
    Merge ranked results from multiple retrievers.

    Duplicate chunks are identified using chunk_id.
    """

    fused_scores: dict[str, float] = {}
    documents_by_id: dict[str, Document] = {}

    for results, weight in zip(result_lists, weights):
        for rank, document in enumerate(results, start=1):
            chunk_id = document.metadata["chunk_id"]

            documents_by_id[chunk_id] = document

            previous_score = fused_scores.get(chunk_id, 0.0)
            rank_score = weight / (rank_constant + rank)

            fused_scores[chunk_id] = previous_score + rank_score

    ranked_chunk_ids = sorted(
        fused_scores,
        key=lambda chunk_id: fused_scores[chunk_id],
        reverse=True,
    )

    ranked_results = [
        (
            documents_by_id[chunk_id],
            fused_scores[chunk_id],
        )
        for chunk_id in ranked_chunk_ids
    ]

    # A small deterministic authority boost keeps official policies and
    # tables above blogs when their relevance is otherwise nearly equal.
    boosted_results = [
        (
            document,
            score + (
                PRIMARY_AUTHORITY_BOOST
                if document.metadata.get("authority") == "primary"
                else 0.0
            ),
        )
        for document, score in ranked_results
    ]

    return sorted(
        boosted_results,
        key=lambda item: item[1],
        reverse=True,
    )



def hybrid_retrieve(
    question: str,
    chunks: list[Document],
    vector_store,
    dense_k: int = DENSE_TOP_K,
    bm25_k: int = BM25_TOP_K,
    final_k: int = FINAL_TOP_K,
    topic: str | None = None,
    department: str | None = None,
) -> list[tuple[Document, float]]:

    with tracer.start_as_current_span(
        "aeronova.hybrid_retrieve"
    ) as span:
        year = extract_policy_year(question)
        requested_date = extract_requested_date(question)
        if topic is None and department is None:
           topic = infer_question_topic(question)

        span.set_attribute(
            "rag.question_year",
            year or 0,
        )
        span.set_attribute(
            "rag.question_topic",
            topic or "unclassified",
        )
        span.set_attribute(
            "rag.dense_top_k",
            dense_k,
        )
        span.set_attribute(
            "rag.bm25_top_k",
            bm25_k,
        )

        print(f"Retrieval department: {department}; topic: {topic}")
        span.set_attribute("rag.department", department or "unfiltered")

        # ---------------------------------
        # Dense retrieval from Qdrant
        # ---------------------------------

        qdrant_filter = create_qdrant_filter(
            year=year,
            topic=topic,
            department=department,
        )

        with measure_step("Dense search (embedding + Qdrant)"):
            dense_results_with_scores = (
                vector_store.similarity_search_with_score(
                    query=question,
                    k=dense_k,
                    filter=qdrant_filter,
                )
            )

        dense_documents = [
            document
            for document, similarity_score
            in dense_results_with_scores
            if document_is_valid_for_date(
                document,
                requested_date,
            )
        ]

        span.set_attribute(
            "rag.dense_result_count",
            len(dense_documents),
        )

        span.set_attribute(
            "rag.dense_chunk_ids",
            ",".join(
                document.metadata.get(
                    "chunk_id",
                    "unknown",
                )
                for document in dense_documents
            ),
        )

        # ---------------------------------
        # Keyword retrieval with BM25
        # ---------------------------------

        eligible_chunks = filter_chunks_by_year(
            chunks=chunks,
            year=year,
        )

        if department is not None:
            eligible_chunks = [chunk for chunk in eligible_chunks
                               if chunk.metadata.get("department") == department]

        if topic is not None:
            eligible_chunks = [
                chunk
                for chunk in eligible_chunks
                if chunk.metadata.get("topic") == topic
            ]

        eligible_chunks = [
            chunk
            for chunk in eligible_chunks
            if document_is_valid_for_date(
                chunk,
                requested_date,
            )
        ]

        print(
            "Eligible BM25 chunks:",
            [
                (
                    chunk.metadata.get("chunk_id"),
                    chunk.metadata.get("topic"),
                )
                for chunk in eligible_chunks
            ],
        )

        with measure_step("BM25 build + search"):
            bm25_documents = []
    
            if eligible_chunks:
                bm25_retriever = (
                    BM25Retriever.from_documents(
                        eligible_chunks
                    )
                )
                bm25_retriever.k = bm25_k
                bm25_documents = (
                    bm25_retriever.invoke(question)
                )

        span.set_attribute(
            "rag.bm25_result_count",
            len(bm25_documents),
        )

        span.set_attribute(
            "rag.bm25_chunk_ids",
            ",".join(
                document.metadata.get(
                    "chunk_id",
                    "unknown",
                )
                for document in bm25_documents
            ),
        )

        # ---------------------------------
        # Reciprocal Rank Fusion
        # ---------------------------------

        fused_results = reciprocal_rank_fusion(
            result_lists=[
                dense_documents,
                bm25_documents,
            ],
            weights=[
                DENSE_WEIGHT,
                BM25_WEIGHT,
            ],
        )

        # Defensive filter: never send a different
        # topic to the answer-generating LLM.
        if topic is not None:
            fused_results = [
                (document, score)
                for document, score in fused_results
                if document.metadata.get("topic") == topic
            ]

        if department is not None:
            fused_results = [(document, score) for document, score in fused_results
                             if document.metadata.get("department") == department]

        final_results = fused_results[:final_k]

        span.set_attribute(
            "rag.final_result_count",
            len(final_results),
        )

        span.set_attribute(
            "rag.result_document_ids",
            ",".join(
                document.metadata.get(
                    "document_id",
                    "unknown",
                )
                for document, score in final_results
            ),
        )

        span.set_attribute(
            "rag.result_chunk_ids",
            ",".join(
                document.metadata.get(
                    "chunk_id",
                    "unknown",
                )
                for document, score in final_results
            ),
        )

        return final_results


def has_primary_evidence(
    results: list[tuple[Document, float]],
) -> bool:
    return any(
        document.metadata.get("authority") == "primary"
        for document, score in results
    )


def create_vector_store() -> QdrantVectorStore:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    client = QdrantClient(
        url=QDRANT_URL
    )

    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Qdrant collection does not exist: "
            f"{COLLECTION_NAME}. Run build_index.py first."
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    return vector_store


def print_result(
    position: int,
    document,
    score: float,
) -> None:
    metadata = document.metadata

    print("\n" + "=" * 70)
    print(f"RESULT {position}")
    print(f"Hybrid RRF score: {score:.6f}")
    print(
        f"Document ID: "
        f"{metadata.get('document_id', 'unknown')}"
    )
    print(
        f"Source: "
        f"{metadata.get('source_file', 'unknown')}"
    )
    print(
        f"Page: "
        f"{metadata.get('page', 'unknown')}"
    )
    print(
        f"Policy year: "
        f"{metadata.get('policy_year', 'unknown')}"
    )
    print(
        f"Authority: "
        f"{metadata.get('authority', 'unknown')}"
    )
    print(
        f"Valid: "
        f"{metadata.get('valid_from', 'unknown')} to "
        f"{metadata.get('valid_to', 'unknown')}"
    )
    print(
        f"Chunk ID: "
        f"{metadata.get('chunk_id', 'unknown')}"
    )
    print("-" * 70)
    print(document.page_content)


def main() -> None:
    from agent.jev_classifier import classify_and_route, routing_handoff_message
    from safety.policy import filter_safe_context

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing from the .env file"
        )

    # Generate exactly the same chunks used by build_index.py.
    # This does not write duplicate vectors to Qdrant.
    chunks = prepare_public_chunks()

    print(f"Chunks available to BM25: {len(chunks)}")

    ensure_index_is_fresh(chunks)

    vector_store = create_vector_store()

    while True:
        question = input(
            "\nAsk AeroNova "
            "(or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("Goodbye")
            break

        if not question:
            continue
        with measure_step("JEV department + scope"):
            decision = classify_and_route(question)
        if not decision.rag_allowed:
            print(routing_handoff_message(decision))
            continue
        results = hybrid_retrieve(
            question=question,
            chunks=chunks,
            vector_store=vector_store,
            final_k=5,
            department=decision.department
        )

        results = filter_safe_context(results)
        if not results:
            year = extract_policy_year(question)

            if year is not None:
                print(
                    "No documents matched policy year "
                    f"{year}."
                )
            else:
                print("No relevant documents found.")

            continue

        if not has_primary_evidence(results):
            print(
                "Secondary material was found, but no applicable "
                "primary policy supports the answer. Refusing to "
                "fall back silently."
            )
            continue

        for position, (document, score) in enumerate(
            results,
            start=1,
        ):
            print_result(
                position=position,
                document=document,
                score=score,
            )


if __name__ == "__main__":
    main()
