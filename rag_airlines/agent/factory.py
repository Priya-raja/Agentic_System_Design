"""Composition root and compatibility exports. Orchestration lives in ChatSession."""

import os
from dataclasses import dataclass
from typing import Any
from dotenv import load_dotenv
from agent.answers import (
    GroundedAnswer,
    generate_answer,
    serialize_results,
    deserialize_results,
    format_context,
    validate_citations,
    validate_measurements,
    extract_measurements,
    print_final_answer,
)
from agent.prompt_manager import PromptVersion, load_answer_prompt
from context.ingest.prepare_documents import prepare_public_chunks
from context.indexers.retrieve import create_vector_store, ensure_index_is_fresh
from config import ANSWER_PROMPT_VERSION


@dataclass(frozen=True)
class AeroNovaAgent:
    prompt: PromptVersion
    chunks: list[Any]
    vector_store: Any


def create_agent() -> AeroNovaAgent:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing")
    prompt = load_answer_prompt(version=ANSWER_PROMPT_VERSION)
    chunks = prepare_public_chunks()
    ensure_index_is_fresh(chunks)
    return AeroNovaAgent(prompt, chunks, create_vector_store())


def main(refund_agent=None) -> None:
    from agent.session import ChatSession
    from agent.cli import print_result, refund_command
    from observability.logging import configure_logging, enable_optional_tracing

    configure_logging()
    enable_optional_tracing()
    session = ChatSession.build(refund_agent=refund_agent)
    print("AeroNova RAG assistant is ready. Answers are grounded and verified.")
    while True:
        try:
            question = input("\nAsk AeroNova (or type 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() == "exit":
            break
        if not question:
            continue
        if question.startswith("/refund"):
            refund_command(question, refund_agent)
            continue
        print_result(session.ask(question))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AeroNova bounded agent")
    parser.add_argument("--demo-refund-customer")
    args = parser.parse_args()
    if args.demo_refund_customer:
        from agent.refund_agent import RefundAgent

        main(RefundAgent(demo=True, customer_id=args.demo_refund_customer))
    else:
        main()

__all__ = [
    "AeroNovaAgent",
    "create_agent",
    "main",
    "GroundedAnswer",
    "generate_answer",
    "serialize_results",
    "deserialize_results",
    "format_context",
    "validate_citations",
    "validate_measurements",
    "extract_measurements",
    "print_final_answer",
]
