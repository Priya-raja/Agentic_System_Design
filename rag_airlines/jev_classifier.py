import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from opentelemetry import trace

from config import (
    JEV_API_URL,
    JEV_MODEL,
    JEV_ROUTE_MIN_CONFIDENCE,
    JEV_TIMEOUT_SECONDS,
    JEV_TOPIC_MIN_CONFIDENCE,
)


load_dotenv()

tracer = trace.get_tracer(__name__)


@dataclass(frozen=True)
class JevRoutingDecision:
    topic: str | None
    topic_confidence: float
    model_tier: str
    model_confidence: float


QUESTIONS = {
    "topic": {
        "type": "choice",
        "instructions": (
            "What is the primary topic of this AeroNova "
            "Airways customer question?"
        ),
        "criteria": {
            "menu": (
                "Meals, food, drinks, menu prices, dietary "
                "meals, catering, or meal preorders."
            ),
            "baggage": (
                "Checked baggage, cabin baggage, luggage, "
                "allowances, excess baggage, dimensions, "
                "sporting equipment, batteries, or power banks."
            ),
            "seating": (
                "Seat categories, pitch, recline, lie-flat "
                "seats, cabin class, or seating facilities."
            ),
            "general": (
                "An airline question that does not clearly "
                "belong to menu, baggage, or seating."
            ),
        },
    },
    "model_tier": {
        "type": "choice",
        "instructions": (
            "How difficult is it to answer this question "
            "accurately from retrieved airline documents?"
        ),
        "criteria": {
            "cheap": (
                "A direct factual lookup, price lookup, simple "
                "list, route-specific lookup, or straightforward "
                "refusal when information is unavailable."
            ),
            "standard": (
                "Requires comparing documents, resolving dates "
                "or policy versions, applying exceptions, "
                "interpreting eligibility, calculations, or "
                "handling several conditions."
            ),
        },
    },
}


def fallback_decision(
    reason: str,
) -> JevRoutingDecision:
    print(
        f"WARNING: {reason}. "
        "Using local topic fallback and standard model."
    )

    return JevRoutingDecision(
        topic=None,
        topic_confidence=0.0,
        model_tier="standard",
        model_confidence=0.0,
    )


def classify_and_route(
    question: str,
) -> JevRoutingDecision:
    api_key = os.getenv("OPENCODE_API_KEY")

    if not api_key:
        return fallback_decision(
            "OPENCODE_API_KEY is missing"
        )

    request_body = {
        "model": JEV_MODEL,
        "state": question,
        "questions": QUESTIONS,
    }

    try:
        with tracer.start_as_current_span(
            "aeronova.jev_classification"
        ) as span:
            response = httpx.post(
                JEV_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=request_body,
                timeout=JEV_TIMEOUT_SECONDS,
            )

            response.raise_for_status()
            response_data = response.json()

            answers = response_data["answers"]

            topic_answer = answers["topic"]
            model_answer = answers["model_tier"]

            topic = topic_answer["choice"]
            topic_confidence = float(
                topic_answer["confidence"]
            )

            model_tier = model_answer["choice"]
            model_confidence = float(
                model_answer["confidence"]
            )

            if (
                topic == "general"
                or topic_confidence
                < JEV_TOPIC_MIN_CONFIDENCE
            ):
                topic = None

            # Never choose the cheap model when
            # JEV is uncertain.
            if (
                model_confidence
                < JEV_ROUTE_MIN_CONFIDENCE
            ):
                model_tier = "standard"

            span.set_attribute(
                "jev.model",
                JEV_MODEL,
            )
            span.set_attribute(
                "jev.topic",
                topic or "unclassified",
            )
            span.set_attribute(
                "jev.topic_confidence",
                topic_confidence,
            )
            span.set_attribute(
                "jev.model_tier",
                model_tier,
            )
            span.set_attribute(
                "jev.model_confidence",
                model_confidence,
            )

            decision = JevRoutingDecision(
                topic=topic,
                topic_confidence=topic_confidence,
                model_tier=model_tier,
                model_confidence=model_confidence,
            )

            print(
                f"JEV topic: {topic or 'unclassified'} "
                f"(confidence={topic_confidence:.3f})"
            )
            print(
                f"JEV model tier: {model_tier} "
                f"(confidence={model_confidence:.3f})"
            )

            return decision

    except httpx.HTTPStatusError as error:
        status_code = error.response.status_code
        response_text = error.response.text[:300]

        return fallback_decision(
            f"JEV returned HTTP {status_code}: "
            f"{response_text}"
        )

    except (
        httpx.HTTPError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        return fallback_decision(
            f"JEV classification failed: {error}"
        )


if __name__ == "__main__":
    test_questions = [
        "What is the menu for DXB to London?",
        (
            "How much checked baggage did Economy Flex "
            "include in November 2025?"
        ),
        (
            "Does the Gold baggage addition apply to "
            "infant group fares?"
        ),
    ]

    for test_question in test_questions:
        print("\nQuestion:", test_question)
        print(classify_and_route(test_question))