from phoenix.otel import register

from config import (
    PHOENIX_ENABLED,
    PHOENIX_ENDPOINT,
    PHOENIX_PROJECT_NAME
)


_tracing_initialized = False


def setup_tracing():
    global _tracing_initialized

    if not PHOENIX_ENABLED:
        return None

    if _tracing_initialized:
        return None

    tracer_provider = register(
        project_name=PHOENIX_PROJECT_NAME,
        endpoint=PHOENIX_ENDPOINT,
        protocol="http/protobuf",
        auto_instrument=True,
        batch=True,
    )

    _tracing_initialized = True

    return tracer_provider