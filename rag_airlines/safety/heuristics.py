"""Deterministic detection and redaction. These signals are not intent judgments."""
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

BLOCK = "BLOCK"
WARN = "WARN"
RETRY = "RETRY"
SECURE = "SECURE"
REVIEW = "REVIEW"
OK = "OK"

PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "booking_reference": re.compile(
        r"(?i:\b(?:booking(?:\s+reference)?|pnr|reference)\s*(?:is\s+)?[:#-]?\s*)"
        r"(?P<value>(?=[A-Za-z0-9]{5,8}\b)(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{5,8}|[A-Z]{6})\b"
    ),
    "ticket_number": re.compile(
        r"(?i:\b(?:ticket|e-ticket)\s*(?:number|no\.?)?\s*(?:is\s+)?[:#-]?\s*)"
        r"(?P<value>\d{3}[- ]?\d{10}|\d{10,14})\b"
    ),
    "passport": re.compile(
        r"(?i:\bpassport\s*(?:number|no\.?)?\s*(?:is\s+)?[:#-]?\s*)"
        r"(?P<value>(?=[A-Za-z0-9]{6,9}\b)(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{6,9})\b"
    ),
    "emirates_id": re.compile(r"(?<!\d)784[- ]?\d{4}[- ]?\d{7}[- ]?\d(?!\d)"),
    "uae_phone": re.compile(r"(?<!\w)(?:(?:\+971|00971)[ -]?|0)?5[024568][ -]?\d{3}[ -]?\d{4}(?!\d)"),
    "india_phone": re.compile(r"(?<!\w)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "payment_card": re.compile(r"(?<!\d)(?:\d{4}[ -]){3}\d{4}(?!\d)|\b\d{13,19}\b"),
    "aadhaar": re.compile(r"\b\d{12}\b"),
    "phone": re.compile(r"\b\d{10}\b"),
}
BAD_WORDS = {"fuck", "shit", "bitch", "bastard", "asshole", "dick", "crap", "bloody", "slut"}
UNSAFE_DOMAINS = ["darkweb", "onion", "tor2web", "malware", "phishing", "clickme", "hacktool", "pirated", "crackdownload"]
INJECTION_PATTERN = re.compile(
    r"\b(?:ignore|override|disregard)\s+(?:all\s+|the\s+|your\s+|previous\s+|prior\s+)*"
    r"(?:instructions|system\s+prompt|safety\s+rules)\b|"
    r"\b(?:reveal|print|show|leak)\s+(?:the\s+|your\s+)*(?:system\s+prompt|api\s+keys?|secrets)\b",
    re.I,
)


@dataclass(frozen=True)
class PIIResult:
    detected: bool
    pii_types: tuple[str, ...]
    redacted_text: str


def detect_and_redact_pii(text: str) -> PIIResult:
    redacted = text or ""
    detected = []
    for kind, pattern in PII_PATTERNS.items():
        if not pattern.search(redacted):
            continue
        detected.append(kind)
        placeholder = f"[REDACTED_{kind.upper()}]"
        def replace(match):
            if 'value' in match.re.groupindex:
                start, end = match.span('value')
                return match.group(0)[:start-match.start()] + placeholder + match.group(0)[end-match.start():]
            return placeholder
        redacted = pattern.sub(replace, redacted)
    return PIIResult(bool(detected), tuple(detected), redacted)


def contains_pii(text: str) -> bool:
    return detect_and_redact_pii(text).detected


def contains_profanity(query: str) -> bool:
    return any(token in BAD_WORDS for token in re.findall(r"[a-zA-Z]+", (query or "").casefold()))


def contains_external_link(text: str) -> bool:
    return bool(re.search(r"https?://[^\s]+|\b(?:[\w-]+\.)+(?:com|org|net|onion|zip|click|xyz|ru|cn)\b", text or "", re.I))


def contains_unsafe_link(text: str) -> bool:
    urls = re.findall(r"https?://[^\s]+|\b(?:[\w-]+\.)+(?:com|org|net|onion|zip|click|xyz|ru|cn)\b", (text or "").casefold())
    for url in urls:
        host = urlsplit(url if '://' in url else 'https://' + url).hostname or ''
        if any(term in host for term in UNSAFE_DOMAINS):
            return True
    return False


def is_tool_not_available(tool_name: str, available_tools: list[str] | None) -> bool:
    return bool(tool_name) and (available_tools is None or tool_name not in available_tools)


def has_incomplete_perception(perception: str | None) -> bool:
    return perception is None or perception.strip().lower() in {"", "none", "no observation", "no context available"} or len(perception.split()) < 3


def detect_model_overload(error_message: str | None) -> dict[str, Any] | None:
    if error_message and re.search(r"\b503\b|service unavailable|model overload|overloaded", error_message, re.I):
        return {"retry": True, "pause_seconds": 20, "reason": "model overload / 503"}
    return None


def handle_overload_and_retry(details: dict[str, Any]) -> dict[str, Any]:
    """Return retry instructions; the caller owns the actual wait."""
    overload = details.get("model_overload")
    if not overload:
        return {"retry": False, "pause_seconds": 0}
    return {"retry": True, "pause_seconds": overload.get("pause_seconds", 20),
            "reason": overload.get("reason", "model overload")}


@dataclass(frozen=True)
class HeuristicAssessment:
    pii: PIIResult
    profanity_detected: bool
    unsafe_link_detected: bool
    action: str
    cache_allowed: bool
    external_link_detected: bool = False
    link_access_allowed: bool = False
    injection_detected: bool = False


def assess_user_input(question: str) -> HeuristicAssessment:
    pii = detect_and_redact_pii(question)
    profanity = contains_profanity(question)
    unsafe_link = contains_unsafe_link(question)
    external_link = contains_external_link(question)
    injection = bool(INJECTION_PATTERN.search(question or ''))
    action = BLOCK if injection else SECURE if pii.detected else REVIEW if unsafe_link else WARN if profanity or external_link else OK
    return HeuristicAssessment(pii, profanity, unsafe_link, action,
                               action in {OK, WARN}, external_link, False, injection)


def is_standalone_insult(text: str) -> bool:
    """Recognize a narrow set of insults without swallowing a service request."""
    normalized = ' '.join((text or '').casefold().strip().split())
    return bool(re.fullmatch(
        r"(?:"
        r"(?:you are|you're|you’re|this (?:assistant|agent|bot) is)\s+"
        r"(?:(?:a|an|the|so|really|totally|completely|bloody|fucking)\s+)*"
        r"(?:useless|stupid|idiotic|incompetent|terrible|an? idiot|asshole|bastard|shit|piece of shit)"
        r"(?:\s+(?:support\s+)?(?:agent|assistant|bot))?"
        r"|fuck you|you suck"
        r")[.!?]*", normalized,
    ))
