import json
import hashlib
from pathlib import Path
import pytesseract
from PIL import Image
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

CORPUS_ROOT = Path(
    "/Users/priyaraja/projects/Agentic_System_Design/"
    "rag_airlines/data"
)
PDF_METADATA_TO_REMOVE = {
    "producer",
    "creator",
    "creationdate",
    "moddate",
    "modDate",
    "creationDate",
    "trapped",
}


def file_freshness_metadata(file_path: Path) -> dict:
    """Return stable metadata used to detect changed source files."""

    content_hash = hashlib.sha256(
        file_path.read_bytes()
    ).hexdigest()

    return {
        "content_hash": content_hash,
        "source_modified_at": file_path.stat().st_mtime,
    }


def load_pdf(file_path: Path) -> list[Document]:
    loader = PyMuPDFLoader(str(file_path))
    documents = loader.load()
    freshness = file_freshness_metadata(file_path)

    for document in documents:

        for key in PDF_METADATA_TO_REMOVE:
            document.metadata.pop(
                key,
                None,
            )
        document.metadata.pop("source", None)
        document.metadata.pop("file_path", None)

        page_index = document.metadata.get(
            "page",
            0,
        )

        document.metadata.update(
            {
                "document_id": file_path.stem,
                "source_file": file_path.name,
                "source_type": "pdf",
                "corpus_folder": file_path.parent.name,
                "page": page_index + 1,
                **freshness,
            }
        )

    return documents

def load_image(file_path: Path) -> list[Document]:
    image = Image.open(file_path)
    freshness = file_freshness_metadata(file_path)

    extracted_text = pytesseract.image_to_string(
        image,
        config="--psm 6",
    ).strip()

    return [
        Document(
            page_content=extracted_text,
            metadata={
                "document_id": file_path.stem,
                "source_file": file_path.name,
                "source_type": "image",
                "corpus_folder": file_path.parent.name,
                "image_path": str(file_path),
                "page": 1,
                **freshness,
            },
        )
    ]
def load_markdown(file_path: Path) -> list[Document]:
    content = file_path.read_text(encoding="utf-8")
    freshness = file_freshness_metadata(file_path)

    return [
        Document(
            page_content=content,
            metadata={
                "document_id": file_path.stem,
                "source_file": file_path.name,
                "source_type": "markdown",
                "corpus_folder": file_path.parent.name,
                **freshness,
            },
        )
    ]
def load_json(file_path: Path) -> list[Document]:
    content = file_path.read_text(encoding="utf-8")
    freshness = file_freshness_metadata(file_path)

    return [
        Document(
            page_content=content,
            metadata={
                "document_id": file_path.stem,
                "source_file": file_path.name,
                "source_type": "json",
                "corpus_folder": file_path.parent.name,
                **freshness,
            },
        )
    ]
def load_jsonl(file_path: Path) -> list[Document]:
    documents = []
    freshness = file_freshness_metadata(file_path)

    with file_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            documents.append(
                Document(
                    page_content=json.dumps(
                        record,
                        ensure_ascii=False,
                    ),
                    metadata={
                        "document_id": record.get(
                            "document_id",
                            record.get(
                                "customer_id",
                                record.get(
                                    "case_id",
                                    f"{file_path.stem}-{line_number}",
                                ),
                            ),
                        ),
                        "source_file": file_path.name,
                        "source_type": "jsonl",
                        "corpus_folder": file_path.parent.name,
                        "line_number": line_number,
                        **freshness,
                    },
                )
            )

    return documents


def load_all_documents() -> list[Document]:
    documents = []

    # rglob searches the root and every folder underneath it.
    for file_path in sorted(CORPUS_ROOT.rglob("*")):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        print(f"Reading: {file_path}")

        try:
            if suffix == ".pdf":
                documents.extend(load_pdf(file_path))

            elif suffix in {".png", ".jpg", ".jpeg"}:
                documents.extend(load_image(file_path))

            elif suffix == ".md":
                documents.extend(load_markdown(file_path))

            elif suffix == ".json":
                documents.extend(load_json(file_path))

            elif suffix == ".jsonl":
                documents.extend(load_jsonl(file_path))

        except Exception as error:
            print(f"Could not read {file_path}: {error}")

    return documents


if __name__ == "__main__":
    if not CORPUS_ROOT.exists():
        raise FileNotFoundError(
            f"Corpus folder does not exist: {CORPUS_ROOT}"
        )

    loaded_documents = load_all_documents()

    print("\n" + "=" * 60)
    print(f"Total LangChain documents: {len(loaded_documents)}")

    for document in loaded_documents[:5]:
        print("\nMetadata:")
        print(document.metadata)

        print("\nContent preview:")
        print(document.page_content[:300])
