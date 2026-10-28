from observability.logging import log_message
from config import (
    ROOT_DIR,
    INDEX_SCHEMA_VERSION,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    EMBEDDING_SIZE,
)
import json
import os
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from context.ingest.prepare_documents import prepare_public_chunks


COLLECTION_NAME = QDRANT_COLLECTION
INDEX_STATE_FILE = ROOT_DIR / "index_state.json"


def build_source_state(chunks) -> dict[str, str]:
    """Map each indexed source file to its SHA-256 content hash."""

    return {chunk.metadata["source_file"]: chunk.metadata["content_hash"] for chunk in chunks}


def main() -> None:
    from observability.logging import configure_logging

    configure_logging()
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing from the .env file")

    # Load, select and chunk the public corpus.
    chunks = prepare_public_chunks()

    indexed_at = datetime.now(timezone.utc).isoformat()

    for chunk in chunks:
        if "content_hash" not in chunk.metadata:
            log_message(
                "Missing content_hash:",
                chunk.metadata.get("source_file"),
                chunk.metadata.get("source_type"),
                chunk.metadata.get("chunk_id"),
            )

    log_message(f"Chunks to index: {len(chunks)}")

    # This model converts text into 1536-dimensional vectors.
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # A full rebuild is intentional for this small corpus. It removes
    # vectors for edited or deleted documents and prevents stale policies
    # from remaining searchable.
    if client.collection_exists(COLLECTION_NAME):
        log_message("Deleting previous collection...")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_SIZE,
            distance=Distance.COSINE,
        ),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    # Qdrant accepts UUIDs as point identifiers.
    qdrant_ids = []

    for chunk in chunks:
        chunk_id = chunk.metadata["chunk_id"]

        qdrant_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk_id,
            )
        )

        qdrant_ids.append(qdrant_id)

    log_message("Creating embeddings and saving chunks...")

    vector_store.add_documents(
        documents=chunks,
        ids=qdrant_ids,
    )

    collection_info = client.get_collection(COLLECTION_NAME)

    index_state = {
        "collection_name": COLLECTION_NAME,
        "schema_version": INDEX_SCHEMA_VERSION,
        "indexed_at": indexed_at,
        "documents": build_source_state(chunks),
    }

    INDEX_STATE_FILE.write_text(
        json.dumps(index_state, indent=2),
        encoding="utf-8",
    )

    log_message("\nIndex created successfully")
    log_message(f"Collection: {COLLECTION_NAME}")
    log_message(f"Qdrant points: {collection_info.points_count}")
    log_message(f"Index state: {INDEX_STATE_FILE.name}")


if __name__ == "__main__":
    main()
