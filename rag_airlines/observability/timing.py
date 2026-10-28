"""Correlated structured step timings."""

import logging
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def measure_step(label: str):
    started = perf_counter()
    try:
        yield
    finally:
        logging.getLogger("aeronova.timing").info(
            "step_completed", extra={"step": label, "ms": round((perf_counter() - started) * 1000, 2)}
        )
