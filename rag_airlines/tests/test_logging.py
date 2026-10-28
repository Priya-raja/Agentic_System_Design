import json
import logging
from observability.logging import JsonFormatter, sanitize_for_log, request_id


def test_logs_redact_nested_identifiers_secrets_and_correlate():
    value = sanitize_for_log(
        {
            "email": "person@example.com",
            "nested": ["booking AN7K2P", "Bearer secret123", "sk-secretvalue"],
            "csrf_token": "secret",
        }
    )
    text = json.dumps(value)
    for private in ["person@example.com", "AN7K2P", "secret123", "sk-secretvalue"]:
        assert private not in text
    token = request_id.set("request-test")
    try:
        record = logging.LogRecord("test", 20, "file", 1, "hello", (), None)
        assert json.loads(JsonFormatter().format(record))["request_id"] == "request-test"
    finally:
        request_id.reset(token)
