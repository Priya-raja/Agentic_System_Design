"""Bounded orchestration of transient model-overload retries."""
import time
from safety.heuristics import detect_model_overload, handle_overload_and_retry


def run_with_overload_retry(operation, *, max_retries: int = 1):
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as error:
            status = getattr(error, 'status_code', None)
            if status is None:
                status = getattr(getattr(error, 'response', None), 'status_code', None)
            details = handle_overload_and_retry({
                'model_overload': detect_model_overload(str(status) if status else str(error)),
            })
            if not details['retry'] or attempt >= max_retries:
                raise
            # Log no error body: it may contain request data.
            print("Model temporarily unavailable; retrying once after 20 seconds.")
            time.sleep(min(20, max(0, details['pause_seconds'])))
