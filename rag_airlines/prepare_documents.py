import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from load_documents import load_all_documents


PUBLIC_FOLDERS = {
    "policies",
    "web",
}

SKIP_FILES = {
    "README.md",
    "manifest.json",
}

ALLOWED_IMAGE_FILES = {
    "IMG-BAG-UPDATE-2026.png",
}


def extract_document_year(
    source_file: str,
    content: str,
) -> int | None:
    """Extract a policy year from the filename or document header."""

    filename_match = re.search(
        r"\b(20\d{2})\b",
        source_file,
    )

    if filename_match:
        return int(filename_match.group())

    # Web documents such as WEB-001.md keep their publication year
    # in the front matter rather than in the filename.
    content_match = re.search(
        r"\b(20\d{2})\b",
        content[:500],
    )

    if content_match:
        return int(content_match.group())

    return None


def clean_ocr_text(text: str) -> str:
    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        # Remove OCR borders such as | and repeated punctuation.
        line = re.sub(r"^[|.=,\"]+\s*", "", line)
        line = re.sub(r"\s*[|.=,\"]+$", "", line)

        # Replace repeated spaces with one space.
        line = re.sub(r"\s+", " ", line)

        # Ignore empty lines.
        if not line:
            continue

        # Ignore lines containing only symbols.
        if not re.search(r"[A-Za-z0-9]", line):
            continue

        # Ignore common noise produced by this sample image.
        if line.lower() in {"ee"}:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def select_public_documents(
    documents: list[Document],
) -> list[Document]:
    selected_documents = []

    for document in documents:
        metadata = document.metadata

        folder = metadata.get("corpus_folder")
        source_file = metadata.get("source_file")
        source_type = metadata.get("source_type")

        # Skip README and manifest.
        if source_file in SKIP_FILES:
            continue

        # Keep only policies and web articles.
        if folder not in PUBLIC_FOLDERS:
            continue

        # For now, only index the policy notice image.
        # The seat and meal pictures will enter the image index later.
        if (
            source_type == "image"
            and source_file not in ALLOWED_IMAGE_FILES
        ):
            continue

        content = document.page_content.strip()

        if source_type == "image":
            content = clean_ocr_text(content)

        if not content:
            continue

        metadata_copy = metadata.copy()
        policy_year = extract_document_year(
            source_file=source_file,
            content=content,
        )

        if policy_year is not None:
            metadata_copy["policy_year"] = policy_year

        selected_documents.append(
            Document(
                page_content=content,
                metadata=metadata_copy,
            )
        )

    return selected_documents


def split_documents(
    documents: list[Document],
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(documents)

    for position, chunk in enumerate(chunks):
        document_id = chunk.metadata.get(
            "document_id",
            "unknown",
        )

        page = chunk.metadata.get("page", 1)

        chunk.metadata["chunk_id"] = (
            f"{document_id}-page-{page}-chunk-{position}"
        )

    return chunks


def prepare_public_chunks() -> list[Document]:
    all_documents = load_all_documents()

    public_documents = select_public_documents(
        all_documents
    )

    chunks = split_documents(
        public_documents
    )

    return chunks


def main() -> None:
    chunks = prepare_public_chunks()

    print("\n" + "=" * 60)
    print(f"Created chunks: {len(chunks)}")

    for chunk in chunks[:5]:
        print("\n" + "=" * 60)
        print("Metadata:")
        print(chunk.metadata)

        print("\nContent:")
        print(chunk.page_content)


if __name__ == "__main__":
    main()
