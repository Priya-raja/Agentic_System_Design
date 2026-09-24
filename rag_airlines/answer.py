import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from prepare_documents import prepare_public_chunks
from retrieve import (
    create_vector_store,
    ensure_index_is_fresh,
    extract_policy_year,
    has_primary_evidence,
    hybrid_retrieve,
)
from config import (
    ANSWER_MODEL,
    ANSWER_TEMPERATURE,
    ANSWER_PROMPT_VERSION,
    FINAL_TOP_K,
)
from prompt_manager import (
    PromptVersion,
    load_answer_prompt,
)


class GroundedAnswer(BaseModel):
    answerable: bool = Field(
        description=(
            "True only when the supplied context directly "
            "supports the answer."
        )
    )

    answer: str = Field(
        description=(
            "A concise answer based only on the supplied context."
        )
    )

    citation_chunk_ids: list[str] = Field(
        description=(
            "Chunk IDs that directly support the answer."
        )
    )

    refusal_reason: str | None = Field(
        default=None,
        description=(
            "Why the question cannot be answered safely."
        ),
    )



def format_context(results) -> str:
    context_sections = []

    for document, score in results:
        metadata = document.metadata

        section = f"""
CHUNK ID: {metadata.get("chunk_id")}
DOCUMENT ID: {metadata.get("document_id")}
SOURCE: {metadata.get("source_file")}
PAGE: {metadata.get("page", "unknown")}
AUTHORITY: {metadata.get("authority", "unknown")}
STATUS: {metadata.get("status", "unknown")}
VALID FROM: {metadata.get("valid_from", "unknown")}
VALID TO: {metadata.get("valid_to", "unknown")}

CONTENT:
{document.page_content}
""".strip()

        context_sections.append(section)

    return "\n\n---\n\n".join(context_sections)


def validate_citations(
    response: GroundedAnswer,
    results,
) -> list[str]:
    retrieved_chunk_ids = {
        document.metadata.get("chunk_id")
        for document, score in results
    }

    errors = []

    if response.answerable and not response.citation_chunk_ids:
        errors.append(
            "The answer contains no citations."
        )

    for chunk_id in response.citation_chunk_ids:
        if chunk_id not in retrieved_chunk_ids:
            errors.append(
                f"Invented or unavailable citation: {chunk_id}"
            )

    return errors


def extract_measurements(text: str) -> set[str]:
    pattern = (
        r"\b\d+(?:\.\d+)?\s*"
        r"(?:kg|cm|aed|hours?|days?)\b"
    )

    matches = re.findall(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    return {
        re.sub(r"\s+", "", match.lower())
        for match in matches
    }


def validate_measurements(
    response: GroundedAnswer,
    results,
) -> list[str]:
    documents_by_chunk_id = {
        document.metadata.get("chunk_id"): document
        for document, score in results
    }

    cited_documents = [
        documents_by_chunk_id[chunk_id]
        for chunk_id in response.citation_chunk_ids
        if chunk_id in documents_by_chunk_id
    ]

    evidence = "\n".join(
        document.page_content
        for document in cited_documents
    )

    answer_measurements = extract_measurements(
        response.answer
    )

    evidence_measurements = extract_measurements(
        evidence
    )

    unsupported = (
        answer_measurements - evidence_measurements
    )

    return sorted(unsupported)


def generate_answer(
    question: str,
    results,
    prompt: PromptVersion,
) -> GroundedAnswer:
    context = format_context(results)

    llm = ChatOpenAI(
        model=ANSWER_MODEL,
        temperature=ANSWER_TEMPERATURE,
    )

    structured_llm = llm.with_structured_output(
        GroundedAnswer
    )

    user_prompt = f"""
QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Return a grounded answer with the exact chunk IDs
that support it.
"""

    return structured_llm.invoke(
        [
            {
                "role": "system",
                "content": prompt.content,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
    )


def print_final_answer(
    response: GroundedAnswer,
    results,
) -> None:
    if not response.answerable:
        print(
            "\nAeroNova answer: "
            "I cannot confirm an answer from the "
            "available applicable documents."
        )

        if response.refusal_reason:
            print(
                f"Reason: {response.refusal_reason}"
            )

        return

    documents_by_chunk_id = {
        document.metadata.get("chunk_id"): document
        for document, score in results
    }

    print("\n" + "=" * 70)
    print("AERONOVA ANSWER")
    print("=" * 70)
    print(response.answer)

    print("\nSources:")

    for chunk_id in response.citation_chunk_ids:
        document = documents_by_chunk_id[chunk_id]
        metadata = document.metadata

        print(
            f"- {metadata.get('document_id')} | "
            f"{metadata.get('source_file')} | "
            f"Page {metadata.get('page', 'unknown')} | "
            f"Chunk {chunk_id}"
        )


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing from .env"
        )

    prompt = load_answer_prompt(
        version=ANSWER_PROMPT_VERSION
    )

    print(
        f"Using prompt: "
        f"{prompt.prompt_id}@{prompt.version}"
    )
    print(
        f"Prompt hash: "
        f"{prompt.content_hash[:12]}"
    )

    chunks = prepare_public_chunks()

    # Prevent fresh BM25 chunks from being mixed with
    # stale vectors in Qdrant.
    ensure_index_is_fresh(chunks)

    vector_store = create_vector_store()

    print("AeroNova RAG assistant is ready.")

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
            final_k=FINAL_TOP_K,
        )

        if not results:
            requested_year = extract_policy_year(
                question
            )

            if requested_year:
                print(
                    "\nAeroNova answer: "
                    f"No applicable policy was found for "
                    f"{requested_year}."
                )
            else:
                print(
                    "\nAeroNova answer: "
                    "No relevant policy was found."
                )

            continue

        if not has_primary_evidence(results):
            print(
                "\nAeroNova answer: "
                "Secondary information was found, but no "
                "applicable primary policy supports it."
            )
            continue

        response = generate_answer(
            question=question,
            results=results,
            prompt=prompt,
        )

        citation_errors = validate_citations(
            response=response,
            results=results,
        )

        if citation_errors:
            print(
                "\nThe generated answer failed "
                "citation validation."
            )

            for error in citation_errors:
                print(f"- {error}")

            continue

        unsupported_measurements = (
            validate_measurements(
                response=response,
                results=results,
            )
        )

        if unsupported_measurements:
            print(
                "\nThe generated answer contained "
                "unsupported measurements:"
            )

            for value in unsupported_measurements:
                print(f"- {value}")

            print(
                "The answer was blocked to prevent "
                "hallucination."
            )
            continue

        print_final_answer(
            response=response,
            results=results,
        )


if __name__ == "__main__":
    main()
