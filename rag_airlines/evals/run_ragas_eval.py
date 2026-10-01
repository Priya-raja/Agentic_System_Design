import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI


# ---------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALS_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Temporary RAGAS 0.4.3 compatibility fix
# Must run before importing ragas.
# ---------------------------------------------------------

from evals.ragas_compat import apply_ragas_compatibility

apply_ragas_compatibility()


# ---------------------------------------------------------
# RAGAS imports: current Collections API
# ---------------------------------------------------------

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    ContextPrecision,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)


# ---------------------------------------------------------
# AeroNova imports
# ---------------------------------------------------------

from answer import (
    GroundedAnswer,
    generate_answer,
    select_answer_model,
    validate_citations,
    validate_measurements,
)
from config import ANSWER_PROMPT_VERSION, FINAL_TOP_K
from jev_classifier import classify_and_route
from prepare_documents import prepare_public_chunks
from prompt_manager import load_answer_prompt
from retrieve import (
    create_vector_store,
    ensure_index_is_fresh,
    has_primary_evidence,
    hybrid_retrieve,
)


GOLDEN_DATASET_FILE = EVALS_DIR / "golden_dataset_v1.jsonl"
RESULTS_DIR = EVALS_DIR / "results"

EVALUATOR_MODEL = "gpt-4.1-mini"


def load_golden_dataset(path: Path) -> list[dict[str, Any]]:
    """Load adjacent JSON records in compact or multi-line form."""

    content = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    cases: list[dict[str, Any]] = []
    position = 0

    while position < len(content):
        while (
            position < len(content)
            and content[position].isspace()
        ):
            position += 1

        if position >= len(content):
            break

        try:
            case, position = decoder.raw_decode(
                content,
                position,
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                "Invalid JSON at "
                f"line {error.lineno}, column {error.colno}: "
                f"{error.msg}"
            ) from error

        if not isinstance(case, dict):
            line_number = (
                content.count("\n", 0, position) + 1
            )
            raise ValueError(
                "Each golden dataset record must be a JSON "
                f"object (record ending on line {line_number})."
            )

        cases.append(case)

    return cases


def create_abstention(reason: str) -> GroundedAnswer:
    """Create the same structured refusal used by the application."""

    return GroundedAnswer(
        answerable=False,
        answer=(
            "I cannot confirm an answer from the available "
            "applicable documents."
        ),
        citation_chunk_ids=[],
        refusal_reason=reason,
    )


def extract_retrieval_data(
    results,
) -> tuple[list[str], list[str], list[str]]:
    """Extract contexts, chunk IDs, and document IDs in rank order."""

    retrieved_contexts = []
    retrieved_chunk_ids = []
    retrieved_document_ids = []

    for document, score in results:
        retrieved_contexts.append(document.page_content)

        retrieved_chunk_ids.append(
            document.metadata.get("chunk_id", "unknown")
        )

        retrieved_document_ids.append(
            document.metadata.get("document_id", "unknown")
        )

    return (
        retrieved_contexts,
        retrieved_chunk_ids,
        retrieved_document_ids,
    )


def calculate_expected_document_recall(
    expected_document_ids: list[str],
    retrieved_document_ids: list[str],
) -> float | None:
    """
    Deterministic retrieval check.

    Returns the fraction of expected document IDs found in retrieval.
    """

    if not expected_document_ids:
        return None

    expected = set(expected_document_ids)
    retrieved = set(retrieved_document_ids)

    return len(expected & retrieved) / len(expected)


def safe_metric_score(
    scorer,
    metric_name: str,
    **kwargs,
) -> tuple[float | None, str | None]:
    """
    Run one RAGAS metric without terminating the entire evaluation
    if one judge call fails.
    """

    try:
        result = scorer.score(**kwargs)
        value = float(result.value)

        if math.isnan(value):
            return None, f"{metric_name} returned NaN"

        return value, None

    except Exception as error:
        return None, f"{metric_name}: {type(error).__name__}: {error}"


def evaluate_answerable_case(
    case: dict[str, Any],
    response: GroundedAnswer,
    retrieved_contexts: list[str],
    scorers: dict[str, Any],
) -> tuple[dict[str, float | None], list[str]]:
    """Run the four RAGAS metrics for an answerable golden case."""

    metric_errors = []

    # If the system incorrectly refuses an answerable question,
    # it should not disappear from the averages.
    if not response.answerable or not retrieved_contexts:
        return (
            {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "faithfulness": 0.0,
                "factual_correctness": 0.0,
            },
            metric_errors,
        )

    context_precision, error = safe_metric_score(
        scorers["context_precision"],
        "context_precision",
        user_input=case["user_input"],
        reference=case["reference"],
        retrieved_contexts=retrieved_contexts,
    )

    if error:
        metric_errors.append(error)

    context_recall, error = safe_metric_score(
        scorers["context_recall"],
        "context_recall",
        user_input=case["user_input"],
        reference=case["reference"],
        retrieved_contexts=retrieved_contexts,
    )

    if error:
        metric_errors.append(error)

    faithfulness, error = safe_metric_score(
        scorers["faithfulness"],
        "faithfulness",
        user_input=case["user_input"],
        response=response.answer,
        retrieved_contexts=retrieved_contexts,
    )

    if error:
        metric_errors.append(error)

    factual_correctness, error = safe_metric_score(
        scorers["factual_correctness"],
        "factual_correctness",
        response=response.answer,
        reference=case["reference"],
    )

    if error:
        metric_errors.append(error)

    return (
        {
            "context_precision": context_precision,
            "context_recall": context_recall,
            "faithfulness": faithfulness,
            "factual_correctness": factual_correctness,
        },
        metric_errors,
    )


def evaluate_case(
    case: dict[str, Any],
    chunks,
    vector_store,
    prompt,
    scorers: dict[str, Any],
) -> dict[str, Any]:
    """Run one golden case through the complete AeroNova pipeline."""

    started_at = time.perf_counter()
    question = case["user_input"]

    routing = classify_and_route(question)
    model_name = select_answer_model(routing)

    retrieval_started_at = time.perf_counter()

    results = hybrid_retrieve(
        question=question,
        chunks=chunks,
        vector_store=vector_store,
        final_k=FINAL_TOP_K,
        topic=routing.topic,
    )

    retrieval_latency_ms = round(
        (time.perf_counter() - retrieval_started_at) * 1000,
        2,
    )

    (
        retrieved_contexts,
        retrieved_chunk_ids,
        retrieved_document_ids,
    ) = extract_retrieval_data(results)

    generation_started_at = time.perf_counter()

    if not results:
        response = create_abstention(
            "No applicable documents were retrieved."
        )

    elif not has_primary_evidence(results):
        response = create_abstention(
            "No applicable primary source supports the answer."
        )

    else:
        response = generate_answer(
            question=question,
            results=results,
            prompt=prompt,
            model_name=model_name,
        )

    generation_latency_ms = round(
        (time.perf_counter() - generation_started_at) * 1000,
        2,
    )

    citation_errors = validate_citations(response, results)

    measurement_errors = []

    if response.answerable:
        measurement_errors = validate_measurements(
            response,
            results,
        )

    actual_abstention = not response.answerable
    expected_abstention = case["should_abstain"]

    abstention_correct = (
        actual_abstention == expected_abstention
    )

    expected_document_recall = (
        calculate_expected_document_recall(
            expected_document_ids=case["expected_document_ids"],
            retrieved_document_ids=retrieved_document_ids,
        )
    )

    # Correct abstentions are evaluated using the deterministic
    # abstention check, not normal answer metrics.
    if expected_abstention:
        ragas_scores = {
            "context_precision": None,
            "context_recall": None,
            "faithfulness": None,
            "factual_correctness": None,
        }
        metric_errors = []

    else:
        ragas_scores, metric_errors = evaluate_answerable_case(
            case=case,
            response=response,
            retrieved_contexts=retrieved_contexts,
            scorers=scorers,
        )

    total_latency_ms = round(
        (time.perf_counter() - started_at) * 1000,
        2,
    )

    return {
        **case,
        "response": response.answer,
        "answerable": response.answerable,
        "refusal_reason": response.refusal_reason,
        "citation_chunk_ids": response.citation_chunk_ids,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_document_ids": retrieved_document_ids,
        "retrieved_contexts": retrieved_contexts,
        "expected_document_recall": expected_document_recall,
        "actual_abstention": actual_abstention,
        "abstention_correct": abstention_correct,
        "jev_topic": routing.topic,
        "jev_topic_confidence": routing.topic_confidence,
        "jev_model_tier": routing.model_tier,
        "jev_model_confidence": routing.model_confidence,
        "answer_model": model_name,
        "citation_errors": citation_errors,
        "measurement_errors": measurement_errors,
        "metric_errors": metric_errors,
        "metrics": ragas_scores,
        "retrieval_latency_ms": retrieval_latency_ms,
        "generation_latency_ms": generation_latency_ms,
        "total_latency_ms": total_latency_ms,
    }


def average_available(
    records: list[dict[str, Any]],
    metric_name: str,
) -> float | None:
    """Average non-null values for one RAGAS metric."""

    values = [
        record["metrics"][metric_name]
        for record in records
        if record["metrics"][metric_name] is not None
    ]

    if not values:
        return None

    return round(mean(values), 4)


def build_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build aggregate scores for the run."""

    document_recall_values = [
        record["expected_document_recall"]
        for record in records
        if record["expected_document_recall"] is not None
    ]

    abstention_values = [
        1.0 if record["abstention_correct"] else 0.0
        for record in records
    ]

    return {
        "total_cases": len(records),
        "answerable_cases": sum(
            not record["should_abstain"]
            for record in records
        ),
        "abstention_cases": sum(
            record["should_abstain"]
            for record in records
        ),
        "context_precision": average_available(
            records,
            "context_precision",
        ),
        "context_recall": average_available(
            records,
            "context_recall",
        ),
        "faithfulness": average_available(
            records,
            "faithfulness",
        ),
        "factual_correctness": average_available(
            records,
            "factual_correctness",
        ),
        "expected_document_recall": (
            round(mean(document_recall_values), 4)
            if document_recall_values
            else None
        ),
        "abstention_accuracy": (
            round(mean(abstention_values), 4)
            if abstention_values
            else None
        ),
        "average_total_latency_ms": round(
            mean(
                record["total_latency_ms"]
                for record in records
            ),
            2,
        ),
        "cases_with_citation_errors": sum(
            bool(record["citation_errors"])
            for record in records
        ),
        "cases_with_measurement_errors": sum(
            bool(record["measurement_errors"])
            for record in records
        ),
        "cases_with_metric_errors": sum(
            bool(record["metric_errors"])
            for record in records
        ),
    }


def save_jsonl(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    """Write detailed per-case results."""

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N golden cases.",
    )

    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Run one specific golden case.",
    )

    args = parser.parse_args()

    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing from .env"
        )

    golden_cases = load_golden_dataset(
        GOLDEN_DATASET_FILE
    )

    if args.case_id:
        golden_cases = [
            case
            for case in golden_cases
            if case["case_id"] == args.case_id
        ]

        if not golden_cases:
            raise ValueError(
                f"Unknown case ID: {args.case_id}"
            )

    elif args.limit is not None:
        golden_cases = golden_cases[: args.limit]

    print(f"Golden cases selected: {len(golden_cases)}")

    # Build application resources only once.
    chunks = prepare_public_chunks()
    ensure_index_is_fresh(chunks)
    vector_store = create_vector_store()

    prompt = load_answer_prompt(
        version=ANSWER_PROMPT_VERSION
    )

    # Build evaluator resources only once.
    evaluator_client = AsyncOpenAI()

    evaluator_llm = llm_factory(
        EVALUATOR_MODEL,
        client=evaluator_client,
    )

    scorers = {
        "context_precision": ContextPrecision(
            llm=evaluator_llm
        ),
        "context_recall": ContextRecall(
            llm=evaluator_llm
        ),
        "faithfulness": Faithfulness(
            llm=evaluator_llm
        ),
        "factual_correctness": FactualCorrectness(
            llm=evaluator_llm,
            mode="f1",
        ),
    }

    records = []

    for position, case in enumerate(
        golden_cases,
        start=1,
    ):
        print(
            f"\n[{position}/{len(golden_cases)}] "
            f"{case['case_id']}"
        )
        print(case["user_input"])

        try:
            record = evaluate_case(
                case=case,
                chunks=chunks,
                vector_store=vector_store,
                prompt=prompt,
                scorers=scorers,
            )

            records.append(record)

            print(
                "Precision:",
                record["metrics"]["context_precision"],
            )
            print(
                "Recall:",
                record["metrics"]["context_recall"],
            )
            print(
                "Faithfulness:",
                record["metrics"]["faithfulness"],
            )
            print(
                "Factual correctness:",
                record["metrics"]["factual_correctness"],
            )
            print(
                "Abstention correct:",
                record["abstention_correct"],
            )
            print(
                "Latency:",
                f"{record['total_latency_ms']} ms",
            )

        except Exception as error:
            print(
                "CASE FAILED:",
                type(error).__name__,
                str(error),
            )

            records.append(
                {
                    **case,
                    "pipeline_error": (
                        f"{type(error).__name__}: {error}"
                    ),
                    "metrics": {
                        "context_precision": None,
                        "context_recall": None,
                        "faithfulness": None,
                        "factual_correctness": None,
                    },
                    "expected_document_recall": None,
                    "abstention_correct": False,
                    "citation_errors": [],
                    "measurement_errors": [],
                    "metric_errors": [],
                    "total_latency_ms": 0.0,
                }
            )

    summary = build_summary(records)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    details_file = (
        RESULTS_DIR
        / f"ragas_run_{timestamp}.jsonl"
    )

    summary_file = (
        RESULTS_DIR
        / f"ragas_summary_{timestamp}.json"
    )

    save_jsonl(records, details_file)

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("AERONOVA RAGAS SUMMARY")
    print("=" * 70)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nDetailed results: {details_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
