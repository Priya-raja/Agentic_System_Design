"""Combine local checks with JEV intent signals and guard context/output."""
from dataclasses import dataclass
from safety.heuristics import (BLOCK, OK, REVIEW, SECURE, WARN, PIIResult,
                               assess_user_input, contains_pii, contains_unsafe_link,
                               INJECTION_PATTERN)


class SafetyViolation(ValueError):
    """Safe-to-display failure; never include the original sensitive content."""


@dataclass(frozen=True)
class SafetyDecision:
    action: str
    cache_allowed: bool
    reason: str


def decide_safety_action(*, local_pii: PIIResult, jev_pii_probability: float,
                         harmful_probability: float, risk_category: str,
                         risk_confidence: float = 1.0) -> SafetyDecision:
    if risk_category in {'prompt_injection', 'privacy_exfiltration', 'fraud',
                         'dangerous_goods_evasion', 'threatening_harm'} and harmful_probability >= .65:
        return SafetyDecision(BLOCK, False, "This request contains unsafe instructions or intent.")
    if local_pii.detected or jev_pii_probability >= .65:
        return SafetyDecision(SECURE, False, "Please use a secure service channel for requests containing personal identifiers.")
    if risk_category != 'safe' or risk_confidence < .65:
        return SafetyDecision(REVIEW, False, "This request needs safety review or clarification.")
    return SafetyDecision(OK, True, "Safety checks passed.")


def ensure_safe_output(response, results) -> None:
    texts = [response.answer, response.refusal_reason or '']
    texts.extend(document.page_content for document, _ in results)
    texts.extend(value for document, _ in results for value in document.metadata.values() if isinstance(value, str))
    if any(contains_pii(text) for text in texts):
        raise SafetyViolation("The response or its sources contain personal identifiers and cannot be displayed or cached.")
    if any(contains_unsafe_link(text) or INJECTION_PATTERN.search(text) for text in texts):
        raise SafetyViolation("The response or its sources contain unsafe content and cannot be displayed or cached.")


def filter_safe_context(results):
    """Exclude sensitive or injected chunks rather than send them to the model."""
    return [(document, score) for document, score in results
            if assess_user_input(document.page_content).action not in {BLOCK, SECURE, REVIEW}
            and cache_payload_allowed(document.metadata)]


def cache_payload_allowed(payload) -> bool:
    """Check string values, not serialized timestamps or numeric metadata."""
    if isinstance(payload, str):
        return (not contains_pii(payload) and not contains_unsafe_link(payload)
                and not INJECTION_PATTERN.search(payload))
    if isinstance(payload, dict):
        return all(cache_payload_allowed(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return all(cache_payload_allowed(value) for value in payload)
    return True
