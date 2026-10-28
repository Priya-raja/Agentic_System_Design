"""Optional Phoenix tracing. Collector availability must never gate an answer."""
import logging
from urllib.parse import urlsplit
import httpx
from opentelemetry import trace
from config import PHOENIX_ENABLED, PHOENIX_ENDPOINT, PHOENIX_PROJECT_NAME, PHOENIX_PUBLIC_URL

log = logging.getLogger("aeronova.tracing")
_tracing_initialized = False
_setup_attempted = False


def setup_tracing():
    global _tracing_initialized, _setup_attempted
    if not PHOENIX_ENABLED or _setup_attempted:
        return None
    _setup_attempted = True
    try:
        from phoenix.otel import register
        # register(batch=True) does not test connectivity: export happens later.
        collector = PHOENIX_ENDPOINT.removesuffix("/v1/traces").rstrip("/")
        httpx.get(f"{collector}/healthz", timeout=1.0).raise_for_status()
        provider = register(
            project_name=PHOENIX_PROJECT_NAME,
            endpoint=PHOENIX_ENDPOINT,
            protocol="http/protobuf",
            auto_instrument=True,
            batch=True,
            verbose=False,
        )
    except Exception as error:
        # Do not log URLs, headers or exception messages that may contain secrets.
        log.warning("Phoenix not detected, tracing disabled. (%s)", type(error).__name__)
        return None
    _tracing_initialized = True
    return provider


def current_trace_metadata():
    """Link the current sampled request, never a cached request's identity."""
    span = trace.get_current_span()
    context = span.get_span_context()
    if not _tracing_initialized or not span.is_recording() or not context.is_valid:
        return {}
    trace_id = format(context.trace_id, "032x")
    metadata = {"trace_id": trace_id}
    url = urlsplit(PHOENIX_PUBLIC_URL)
    if url.scheme in {"http", "https"} and url.netloc and not url.username and not url.password and not url.query and not url.fragment:
        metadata["phoenix_url"] = f"{PHOENIX_PUBLIC_URL.rstrip('/')}/redirects/traces/{trace_id}"
    return metadata
