import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from prepare_documents import prepare_public_chunks


COLLECTION_NAME = "aeronova_public_knowledge"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_SIZE = 1536
INDEX_STATE_FILE = Path(__file__).parent / "index_state.json"


def build_source_state(chunks) -> dict[str, str]:
    """Map each indexed source file to its SHA-256 content hash."""

    return {
        chunk.metadata["source_file"]: chunk.metadata["content_hash"]
        for chunk in chunks
    }


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing from the .env file"
        )

    # Load, select and chunk the public corpus.
    chunks = prepare_public_chunks()

    indexed_at = datetime.now(timezone.utc).isoformat()

    for chunk in chunks:
        if "content_hash" not in chunk.metadata:
            print(
                "Missing content_hash:",
                chunk.metadata.get("source_file"),
                chunk.metadata.get("source_type"),
                chunk.metadata.get("chunk_id"),
            )

    print(f"Chunks to index: {len(chunks)}")

    # This model converts text into 1536-dimensional vectors.
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    client = QdrantClient(
        url=QDRANT_URL
    )

    # A full rebuild is intentional for this small corpus. It removes
    # vectors for edited or deleted documents and prevents stale policies
    # from remaining searchable.
    if client.collection_exists(COLLECTION_NAME):
        print("Deleting previous collection...")
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

    print("Creating embeddings and saving chunks...")

    vector_store.add_documents(
        documents=chunks,
        ids=qdrant_ids,
    )

    collection_info = client.get_collection(
        COLLECTION_NAME
    )

    index_state = {
        "collection_name": COLLECTION_NAME,
        "indexed_at": indexed_at,
        "documents": build_source_state(chunks),
    }

    INDEX_STATE_FILE.write_text(
        json.dumps(index_state, indent=2),
        encoding="utf-8",
    )

    print("\nIndex created successfully")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Qdrant points: {collection_info.points_count}")
    print(f"Index state: {INDEX_STATE_FILE.name}")


if __name__ == "__main__":
    main()
