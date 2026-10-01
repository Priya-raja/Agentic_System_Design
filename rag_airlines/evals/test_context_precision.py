import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas_compat import (
    apply_ragas_compatibility,
)

apply_ragas_compatibility()

from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing from .env"
        )

    client = AsyncOpenAI()

    evaluator_llm = llm_factory(
        "gpt-4.1-mini",
        client=client,
    )

    context_precision = ContextPrecision(
        llm=evaluator_llm
    )

    result = context_precision.score(
        user_input=(
            "How much checked baggage did "
            "Economy Flex include in November 2025?"
        ),
        reference=(
            "Economy Flex included one checked bag "
            "of up to 20 kg in November 2025."
        ),
        retrieved_contexts=[
            (
                "The onboard menu includes saffron "
                "chicken rice for AED 42."
            ),
            (
                "Economy Saver and Economy Flex each "
                "include one checked bag of up to "
                "20 kg."
                        ),
        ],
    )

    print(
        "Context precision:",
        result.value,
    )


if __name__ == "__main__":
    main()