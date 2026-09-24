import re
from datetime import date
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

PRIMARY_DOCUMENT_PREFIXES = (
    "POL-",
    "TAB-",
    "CAT-",
)


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


def determine_authority(
    document_id: str,
    content: str,
) -> str:
    """Classify sources so official policies outrank summaries."""

    authority_match = re.search(
        r"^authority:\s*([a-z]+)\s*$",
        content,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if authority_match:
        return authority_match.group(1).lower()

    if document_id.startswith(PRIMARY_DOCUMENT_PREFIXES):
        return "primary"

    return "secondary"


def determine_status(content: str, policy_year: int | None) -> str:
    """Read an explicit status or infer a simple corpus status."""

    status_match = re.search(
        r"Status:\s*([A-Za-z_-]+)",
        content,
        flags=re.IGNORECASE,
    )

    if status_match:
        return status_match.group(1).lower()

    if policy_year is not None and policy_year < date.today().year:
        return "historical"

    return "active"


def determine_version(content: str) -> str | None:
    version_match = re.search(
        r"Version:\s*([0-9.]+)",
        content,
        flags=re.IGNORECASE,
    )

    return version_match.group(1) if version_match else None


def determine_validity_dates(
    content: str,
    policy_year: int | None,
) -> tuple[str | None, str | None]:
    """Return ISO validity dates, preferring dates printed in the document."""

    iso_range = re.search(
        r"Effective:\s*(\d{4}-\d{2}-\d{2})\s+to\s+"
        r"(\d{4}-\d{2}-\d{2})",
        content,
        flags=re.IGNORECASE,
    )

    if iso_range:
        return iso_range.group(1), iso_range.group(2)

    if policy_year is None:
        return None, None

    # The fictional corpus currently publishes annual policy versions.
    return (
        f"{policy_year}-01-01",
        f"{policy_year}-12-31",
    )


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
        document_id = metadata_copy.get("document_id", "")
        policy_year = extract_document_year(
            source_file=source_file,
            content=content,
        )

        if policy_year is not None:
            metadata_copy["policy_year"] = policy_year

        valid_from, valid_to = determine_validity_dates(
            content=content,
            policy_year=policy_year,
        )

        metadata_copy.update(
            {
                "authority": determine_authority(
                    document_id=document_id,
                    content=content,
                ),
                "status": determine_status(
                    content=content,
                    policy_year=policy_year,
                ),
                "valid_from": valid_from,
                "valid_to": valid_to,
                "version": determine_version(content),
            }
        )

        if document_id == "POL-BAG-2025":
            metadata_copy["superseded_by"] = "POL-BAG-2026"

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
