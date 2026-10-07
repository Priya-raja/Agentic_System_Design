import os
import re
from dataclasses import dataclass

from langchain_qdrant import QdrantVectorStore

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from context.ingest.prepare_documents import prepare_public_chunks
from context.indexers.retrieve import (
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
    NO_CONTEXT_CACHE_TTL_SECONDS,
)
from agent.prompt_manager import (
    PromptVersion,
    load_answer_prompt,
)
from observability.observability import setup_tracing
from observability.timing import measure_step
from agent.jev_classifier import classify_and_route, routing_handoff_message
from agent.router import route_support
from safety.heuristics import assess_user_input, is_standalone_insult, BLOCK, REVIEW, SECURE
from safety.policy import SafetyViolation, decide_safety_action, ensure_safe_output, filter_safe_context
from safety.retry import run_with_overload_retry
from langchain_core.documents import Document
from context.memory.answer_cache import (
    create_answer_cache_key, get_cached_answer, set_cached_answer, get_index_hash,
)
from context.memory.semantic_answer_cache import SemanticAnswerCache, semantic_cache_allowed


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

def serialize_results(results) -> list[dict]:
    """
    Store enough retrieval information to print sources on a
    cache hit without running Qdrant or BM25 again.
    """
    return [
        {
            "page_content": document.page_content,
            "metadata": document.metadata,
            "score": float(score),
        }
        for document, score in results
    ]


def deserialize_results(
    serialized_results: list[dict],
):
    return [
        (
            Document(
                page_content=item["page_content"],
                metadata=item["metadata"],
            ),
            float(item["score"]),
        )
        for item in serialized_results
    ]

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
    model_name: str,
) -> GroundedAnswer:
    input_safety = assess_user_input(question)
    if input_safety.action in {BLOCK, REVIEW, SECURE}:
        raise SafetyViolation("This request requires safety review or a secure service channel.")
    safe_results = filter_safe_context(results)
    if len(safe_results) != len(results):
        raise SafetyViolation("The retrieved context contains sensitive or unsafe content.")
    context = format_context(safe_results)

    llm = ChatOpenAI(
        model=model_name,
        temperature=ANSWER_TEMPERATURE,
        # Group requests with the same system prompt for provider cache routing.
        # The full prompt is still sent; a cache hit is not guaranteed.
        model_kwargs={
            "prompt_cache_key": f"aeronova:{prompt.content_hash}",
        },
    )

    structured_llm = llm.with_structured_output(
        GroundedAnswer,
        include_raw=True,
    )

    user_prompt = f"""
QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Return a grounded answer with the exact chunk IDs
that support it.
"""

    raw_response = run_with_overload_retry(lambda: structured_llm.invoke(
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
    ))

    usage = raw_response["raw"].usage_metadata or {}
    cached_tokens = usage.get("input_token_details", {}).get("cache_read", 0)
    print(f"OpenAI prompt cache: {cached_tokens} cached input tokens")

    if raw_response["parsing_error"] is not None:
        raise raw_response["parsing_error"]

    response = raw_response["parsed"]
    if not isinstance(response, GroundedAnswer):
        response = GroundedAnswer.model_validate(response)
    ensure_safe_output(response, results)
    return response


def print_final_answer(
    response: GroundedAnswer,
    results,
) -> None:
    try:
        ensure_safe_output(response, results)
    except SafetyViolation as error:
        print(f"Safety: {error}")
        return
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


@dataclass(frozen=True)
class AeroNovaAgent:
    """Dependencies shared by the interactive answer pipeline."""

    prompt: PromptVersion
    chunks: list[Document]
    vector_store: QdrantVectorStore


def create_agent() -> AeroNovaAgent:
    """Load the prompt and corpus, verify freshness, and connect retrieval."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing from .env")

    prompt = load_answer_prompt(version=ANSWER_PROMPT_VERSION)
    chunks = prepare_public_chunks()
    ensure_index_is_fresh(chunks)

    return AeroNovaAgent(
        prompt=prompt,
        chunks=chunks,
        vector_store=create_vector_store(),
    )


def main() -> None:
    load_dotenv()
    setup_tracing()
    agent = create_agent()
    prompt = agent.prompt
    chunks = agent.chunks
    vector_store = agent.vector_store
    semantic_cache = SemanticAnswerCache()

    print(
        f"Using prompt: "
        f"{prompt.prompt_id}@{prompt.version}"
    )
    print(
        f"Prompt hash: "
        f"{prompt.content_hash[:12]}"
    )

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

        assessment = assess_user_input(question)
        if assessment.profanity_detected:
            print("Safety warning: profanity detected; continuing with a neutral response.")
        if assessment.external_link_detected:
            print("Safety warning: user-provided links will not be opened.")
        if assessment.action == BLOCK:
            print("Safety: unsafe instructions detected; this request cannot proceed.")
            continue
        if assessment.action == REVIEW:
            print("Safety: this link requires review; please ask without the link.")
            continue
        if assessment.pii.detected:
            print("Safety: personal identifiers detected: " + ", ".join(assessment.pii.pii_types))
        if assessment.cache_allowed and is_standalone_insult(question):
            print("AeroNova answer: I'm here to help. Tell me what went wrong or what you need help with.")
            continue
        # Only the redacted text may leave this function or enter tracing.
        question = assessment.pii.redacted_text

        cache_key = create_answer_cache_key(
            question=question,
            prompt_hash=prompt.content_hash,
        )

        cached_payload = None
        if assessment.cache_allowed:
            with measure_step("Exact cache lookup"):
                cached_payload = get_cached_answer(cache_key)

        if cached_payload is not None:
            print("Answer cache: HIT")
            if cached_payload.get("cache_kind") == "no_context":
                if assess_user_input(cached_payload["message"]).action in {BLOCK, REVIEW, SECURE}:
                    print("Safety: cached response requires review.")
                    continue
                print(f"\nAeroNova answer: {cached_payload['message']}")
                continue

            response = GroundedAnswer.model_validate(
                cached_payload["response"]
            )

            results = deserialize_results(
                cached_payload["results"]
            )

            cached_model = cached_payload.get(
                "model_name",
                "unknown",
            )

            print(f"Cached answer model: {cached_model}")

            print_final_answer(
                response=response,
                results=results,
            )

            continue

        print("Answer cache: MISS" if assessment.cache_allowed else "Answer cache: BYPASS (sensitive input)")

        with measure_step("JEV department + scope"):
            decision = classify_and_route(question)
        safety = decide_safety_action(
            local_pii=assessment.pii,
            jev_pii_probability=decision.pii_probability,
            harmful_probability=decision.harmful_probability,
            risk_category=decision.risk_category,
            risk_confidence=decision.risk_confidence,
        )
        if safety.action in {BLOCK, REVIEW, SECURE}:
            print(f"Safety: {safety.reason}")
            continue
        route = route_support(decision)
        print(f"Destination: {route.destination}")
        if route.destination != "public_rag":
            print(routing_handoff_message(decision))
            continue
        department = decision.department
        cache_metadata = {
            "topic": department,
            "policy_year": extract_policy_year(question),
            "prompt_version": prompt.version,
            "prompt_hash": prompt.content_hash,
            "corpus_version": get_index_hash(),
        }
        allow_semantic_cache = assessment.cache_allowed and safety.cache_allowed and semantic_cache_allowed(question, department)
        if allow_semantic_cache:
            with measure_step("Semantic cache lookup"):
                semantic_payload = semantic_cache.lookup(question=question, **cache_metadata)
            if semantic_payload is not None:
                response = GroundedAnswer.model_validate(semantic_payload["response"])
                results = deserialize_results(semantic_payload["results"])
                if (response.answerable and has_primary_evidence(results)
                        and not validate_citations(response, results)
                        and not validate_measurements(response, results)):
                    try:
                        ensure_safe_output(response, results)
                    except SafetyViolation:
                        print("Safety: cached response requires review.")
                        continue
                    print("Semantic answer cache: HIT")
                    set_cached_answer(cache_key, semantic_payload)
                    print_final_answer(response, results)
                    continue
            print("Semantic answer cache: MISS")

        model_name = ANSWER_MODEL
        print(f"Selected answer model: {model_name}")
        with measure_step("Retrieval total"):
            results = run_with_overload_retry(lambda: hybrid_retrieve(
                question=question,
                chunks=chunks,
                vector_store=vector_store,
                final_k=FINAL_TOP_K,
                department=department,
            ))
        original_result_count = len(results)
        results = filter_safe_context(results)
        if original_result_count and not results:
            print("Safety: no safe retrieved context is available.")
            continue

        if not results:
            requested_year = extract_policy_year(question)
            message = (
                f"No applicable policy was found for {requested_year}."
                if requested_year is not None else "No relevant policy was found."
            )
            # Exact cache only: absence of evidence is not transferable to paraphrases.
            set_cached_answer(
                cache_key=cache_key,
                payload={
                    "cache_kind": "no_context",
                    "message": message,
                    "department": department,
                    "request_scope": decision.request_scope,
                },
                ttl_seconds=NO_CONTEXT_CACHE_TTL_SECONDS,
            )
            print(f"\nAeroNova answer: {message}")
            continue

        if not has_primary_evidence(results):
            print(
                "\nAeroNova answer: "
                "Secondary information was found, but no "
                "applicable primary policy supports it."
            )
            continue

        try:
            with measure_step("Answer generation"):
                response = generate_answer(
                    question=question,
                    results=results,
                    prompt=prompt,
                    model_name=model_name,
                )
            ensure_safe_output(response, results)
        except SafetyViolation as error:
            print(f"Safety: {error}")
            continue
        except Exception as error:
            # Never print provider error bodies, which may contain request content.
            print(f"Answer generation unavailable ({type(error).__name__}); please try again later.")
            continue

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
          
        payload = {
            "response": response.model_dump(mode="json"),
            "results": serialize_results(results),
            "model_name": model_name,
            "prompt_version": prompt.version,
            "department": department,
            "request_scope": decision.request_scope,
        }
        with measure_step("Cache writes"):
            set_cached_answer(cache_key=cache_key, payload=payload)
            if allow_semantic_cache:
                semantic_cache.store(question=question, answer=payload, metadata=cache_metadata)

        print("Validated answer saved to cache.")
        print_final_answer(
            response=response,
            results=results,
        )


if __name__ == "__main__":
    main()
