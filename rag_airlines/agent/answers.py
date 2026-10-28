"""Grounded response schema and synthesis; no interactive orchestration."""

import re
import logging
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from config import ANSWER_TEMPERATURE
from agent.prompt_manager import PromptVersion
from safety.heuristics import assess_user_input, BLOCK, REVIEW, SECURE
from safety.policy import SafetyViolation, ensure_safe_output, filter_safe_context

TICKET_REFUND_GUIDANCE = (
    "For a ticket refund, submit a request through AeroNova's official website or email the "
    "customer-support address listed there. Include your ticket number, booking reference, "
    "passenger name, flight number and travel date, plus a brief explanation. Do not email "
    "payment-card details. This chat cannot check or process ticket refunds."
)
TICKET_REFUND_EMAIL_DRAFT = (
    "Here's a draft you can copy and complete. Send it through the contact form or support email "
    "listed on AeroNova's official website; I cannot send it for you.\n\n"
    "Subject: Ticket refund request\n\n"
    "Dear AeroNova Customer Support,\n\n"
    "I would like to request a refund for my ticket. My details are:\n"
    "- Passenger name: [full name]\n"
    "- Ticket number: [ticket number]\n"
    "- Booking reference: [booking reference]\n"
    "- Flight number and travel date: [flight number and date]\n"
    "- Reason for the request: [brief explanation]\n\n"
    "Please let me know if you need any additional documents and confirm receipt of my request.\n\n"
    "Kind regards,\n[your name]\n\n"
    "Do not include payment-card details."
)


class GroundedAnswer(BaseModel):
    answerable: bool = Field(
        description=("True only when the supplied context directly supports the answer.")
    )

    answer: str = Field(description=("A concise answer based only on the supplied context."))

    citation_chunk_ids: list[str] = Field(description=("Chunk IDs that directly support the answer."))

    refusal_reason: str | None = Field(
        default=None,
        description=("Why the question cannot be answered safely."),
    )


def missing_policy_reason(question, results, existing_reason=None) -> str:
    """Explain the known claims-vs-ticket coverage gap without inventing policy."""
    ids = {document.metadata.get("document_id") for document, _ in results}
    if re.search(r"\brefund\w*\b", question, re.I) and not re.search(
        r"\b(points?|miles|rewards?|loyalty)\b", question, re.I
    ):
        if re.search(r"\b(?:ticket|tickets|flight|fare|cancel(?:led|ed|lation)?)\b", question, re.I):
            return TICKET_REFUND_GUIDANCE
        if not results or ids <= {"POL-CLAIM-2026", "WEB-003"}:
            return TICKET_REFUND_GUIDANCE
    if (
        ids
        and ids <= {"POL-CLAIM-2026", "WEB-003"}
        and re.search(r"\brefund\w*\b", question, re.I)
        and re.search(r"\b(?:flight|ticket|cancelled|canceled|cancellation)\b", question, re.I)
    ):
        return TICKET_REFUND_GUIDANCE
    return (
        existing_reason or "The retrieved documents do not contain the policy needed to answer this question."
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
ROUTE: {metadata.get("route", "not route-specific")}
STATUS: {metadata.get("status", "unknown")}
VALID FROM: {metadata.get("valid_from", "unknown")}
VALID TO: {metadata.get("valid_to", "unknown")}

CONTENT:
{document.page_content}
""".strip()

        context_sections.append(section)

    return "\n\n---\n\n".join(context_sections)


MAX_ANSWER_LENGTH = 2400


def validate_format(response: GroundedAnswer) -> list[str]:
    """Deterministic limits for repetitive or oversized answer text."""
    if not response.answerable:
        return []
    answer = response.answer.strip()
    errors = []
    if len(answer) > MAX_ANSWER_LENGTH:
        errors.append(f"Answer exceeds the {MAX_ANSWER_LENGTH}-character limit.")
    sentences = re.split(r"(?:[.!?]+\s+|\n+)", answer)
    seen = set()
    for sentence in sentences:
        normalized = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", sentence)
        normalized = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", normalized)
        normalized = re.sub(r"[`*_#~]", "", normalized)
        normalized = re.sub(r"[^\w]+", " ", normalized.casefold()).strip()
        if len(normalized.split()) < 4:
            continue
        if normalized in seen:
            errors.append("Answer contains a duplicate sentence.")
            break
        seen.add(normalized)
    return errors


def is_customer_refund_handoff(question: str) -> bool:
    """Recognize explicit personal refund requests, not public refund policy questions."""
    patterns = (
        r"\brefund\s+(?:my|our)\s+(?:tickets?|booking|flight|fare)\b",
        r"\brefund\s+(?:for\s+)?my\s+(?:tickets?|booking|flight|fare)\b",
        r"\b(?:where(?:'s| is)|check(?: on)?|track)\s+my\s+refund\b",
        r"\b(?:connect|transfer|route|speak|talk)\s+(?:me\s+)?(?:to|for|with)\s+(?:an?\s+)?(?:agent\s+)?refund\b",
        r"\b(?:connect|transfer|route|speak|talk)\s+me\s+(?:for|about)\s+(?:a\s+)?refund\b",
        r"\bprocess\s+(?:my\s+)?refund\b",
    )
    return any(re.search(pattern, question, re.I) for pattern in patterns)


def is_ticket_refund_handoff(question: str) -> bool:
    """Route ticket refunds to airline support, except baggage/rewards or EU-law questions."""
    if is_customer_refund_handoff(question):
        return True
    text = question.casefold()
    # EU261 questions are answerable from the regulatory corpus and should
    # continue through public RAG, even when they mention ticket refunds.
    if re.search(
        r"\b(?:eu|ec)\s*261\b|\b261\s*/\s*2004\b|\beu law\b|\bec law\b|\beuropean (?:union|law|regulation)\b",
        text,
    ):
        return False
    if re.search(r"\b(?:baggage|bag|luggage|points?|miles|rewards?|loyalty)\b", text):
        return False
    return bool(re.search(r"\brefund\w*\b", text))


def is_ticket_refund_email_request(question: str) -> bool:
    """Recognize a request to draft (not send) a ticket-refund email."""
    text = question.casefold()
    return bool(
        re.search(r"\b(?:write|draft|compose|prepare)\b", text)
        and re.search(r"\bemail\b", text)
        and is_ticket_refund_handoff(question)
    )

def validate_citations(
    response: GroundedAnswer,
    results,
) -> list[str]:
    retrieved_chunk_ids = {document.metadata.get("chunk_id") for document, score in results}

    errors = []

    if response.answerable and not response.citation_chunk_ids:
        errors.append("The answer contains no citations.")

    for chunk_id in response.citation_chunk_ids:
        if chunk_id not in retrieved_chunk_ids:
            errors.append(f"Invented or unavailable citation: {chunk_id}")

    return errors


def extract_measurements(text: str) -> set[str]:
    from decimal import Decimal

    quantities = set()
    number = r"(?:\d+(?:,\d{3})*(?:\.\d+)?|one|two|three|four|five)"
    number_values = {"one": Decimal(1), "two": Decimal(2), "three": Decimal(3), "four": Decimal(4), "five": Decimal(5)}
    # Normalize equivalent time expressions so legal paraphrases such as
    # "one day" and "24 hours" compare as the same supported quantity.
    time_pattern = re.compile(rf"\b({number})\s*(hours?|days?|minutes?)\b", re.I)
    for value, unit in time_pattern.findall(text):
        amount = number_values.get(value.casefold())
        if amount is None:
            amount = Decimal(value.replace(",", ""))
        unit = unit.casefold()
        factor = Decimal(1440) if unit.startswith("day") else Decimal(60) if unit.startswith("hour") else Decimal(1)
        quantities.add(f"minutes:{(amount * factor).normalize():f}")

    non_time_text = time_pattern.sub(" ", text)
    suffix = r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(kg|cm|aed|eur|points?|months?|%)\b"
    for value, unit in re.findall(suffix, non_time_text, flags=re.I):
        normalized = unit.lower().rstrip("s") if unit.lower() not in {"kg", "cm"} else unit.lower()
        quantities.add(f"{Decimal(value.replace(',', '')).normalize():f}{normalized}")
    for currency, value in re.findall(r"\b(AED|EUR)\s*(\d+(?:,\d{3})*(?:\.\d+)?)", text, flags=re.I):
        unit = currency.lower()
        quantities.add(f"{Decimal(value.replace(',', '')).normalize():f}{unit}")
    for value in re.findall(r"€\s*(\d+(?:,\d{3})*(?:\.\d+)?)", text, flags=re.I):
        quantities.add(f"{Decimal(value.replace(',', '')).normalize():f}eur")
    for value in re.findall(r"\bAED\s*(\d+(?:,\d{3})*(?:\.\d+)?)", text, flags=re.I):
        quantities.add(f"{Decimal(value.replace(',', '')).normalize():f}aed")
    return quantities


def validate_measurements(
    response: GroundedAnswer,
    results,
) -> list[str]:
    documents_by_chunk_id = {document.metadata.get("chunk_id"): document for document, score in results}

    cited_documents = [
        documents_by_chunk_id[chunk_id]
        for chunk_id in response.citation_chunk_ids
        if chunk_id in documents_by_chunk_id
    ]

    evidence = "\n".join(document.page_content for document in cited_documents)

    answer_measurements = extract_measurements(response.answer)

    evidence_measurements = extract_measurements(evidence)

    unsupported = answer_measurements - evidence_measurements

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
        max_retries=0,
        timeout=30,
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
        method="function_calling",
    )

    user_prompt = f"""
QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Return a grounded answer with the exact chunk IDs
that support it.
"""

    raw_response = structured_llm.invoke(
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

    usage = raw_response["raw"].usage_metadata or {}
    cached_tokens = usage.get("input_token_details", {}).get("cache_read", 0)
    logging.getLogger("aeronova.synthesis").info(
        "provider_cache", extra={"details": {"cached_tokens": cached_tokens}}
    )

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
        print("\nAeroNova answer: I cannot confirm an answer from the available applicable documents.")

        if response.refusal_reason:
            print(f"Reason: {response.refusal_reason}")

        return

    documents_by_chunk_id = {document.metadata.get("chunk_id"): document for document, score in results}

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
