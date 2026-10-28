"""JSON logs with request correlation and recursive PII/secret sanitization."""

import json
import logging
import re
from contextvars import ContextVar
from safety.heuristics import detect_and_redact_pii

request_id: ContextVar[str] = ContextVar("request_id", default="system")


def sanitize_for_log(value):
    if isinstance(value, str):
        text = detect_and_redact_pii(value).redacted_text
        return re.sub(
            r"(?i)(?:bearer\s+\S+|sk-[A-Za-z0-9_-]+|(?:api[_ -]?key|token|password)\s*[:=]\s*\S+)",
            "[REDACTED_SECRET]",
            text,
        )
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if key.lower() in {"authorization", "password", "api_key", "csrf_token"}
            else sanitize_for_log(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_for_log(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": request_id.get(),
        }
        for key in ("step", "ms", "details", "message_text"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__
        return json.dumps(sanitize_for_log(payload), default=str)


def configure_logging():
    logger = logging.getLogger("aeronova")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_message(*args, **kwargs):
    logging.getLogger("aeronova.runtime").info("runtime", extra={"message_text": " ".join(map(str, args))})


def enable_optional_tracing():
    import os

    if os.getenv("AERONOVA_ENABLE_OTEL") == "1":
        from observability.observability import setup_tracing

        setup_tracing()
