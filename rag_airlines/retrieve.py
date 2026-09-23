import os
import re

from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from prepare_documents import prepare_public_chunks


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "aeronova_public_knowledge"


def extract_policy_year(question: str) -> int | None:
    """Extract a four-digit year such as 2025 from the question."""

    match = re.search(r"\b(20\d{2})\b", question)

    if match:
        return int(match.group())

    return None


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


def create_qdrant_filter(year: int | None) -> Filter | None:
    """Create the equivalent year filter for Qdrant."""

    if year is None:
        return None

    return Filter(
        must=[
            FieldCondition(
                key="metadata.policy_year",
                match=MatchValue(value=year),
            )
        ]
    )


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

    return [
        (
            documents_by_id[chunk_id],
            fused_scores[chunk_id],
        )
        for chunk_id in ranked_chunk_ids
    ]



def hybrid_retrieve(
    question: str,
    chunks: list[Document],
    vector_store,
    dense_k: int = 10,
    bm25_k: int = 10,
    final_k: int = 5,
) -> list[tuple[Document, float]]:
    year = extract_policy_year(question)

    # -----------------------------
    # Dense retrieval from Qdrant
    # -----------------------------

    qdrant_filter = create_qdrant_filter(year)

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
    ]

    # -----------------------------
    # Keyword retrieval with BM25
    # -----------------------------

    eligible_chunks = filter_chunks_by_year(
        chunks=chunks,
        year=year,
    )

    bm25_documents = []

    if eligible_chunks:
        bm25_retriever = BM25Retriever.from_documents(
            eligible_chunks
        )
        bm25_retriever.k = bm25_k
        bm25_documents = bm25_retriever.invoke(question)

    # -----------------------------
    # Merge both result lists
    # -----------------------------

    fused_results = reciprocal_rank_fusion(
        result_lists=[
            dense_documents,
            bm25_documents,
        ],
        weights=[
            0.6,  # Qdrant
            0.4,  # BM25
        ],
    )

    return fused_results[:final_k]


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
        f"Chunk ID: "
        f"{metadata.get('chunk_id', 'unknown')}"
    )
    print("-" * 70)
    print(document.page_content)


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing from the .env file"
        )

    # Generate exactly the same chunks used by build_index.py.
    # This does not write duplicate vectors to Qdrant.
    chunks = prepare_public_chunks()

    print(f"Chunks available to BM25: {len(chunks)}")

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

        results = hybrid_retrieve(
            question=question,
            chunks=chunks,
            vector_store=vector_store,
            final_k=5,
        )

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
