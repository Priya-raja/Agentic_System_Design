"""Lightweight console timings for diagnosing request latency."""

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def measure_step(label: str):
    started = perf_counter()
    try:
        yield
    finally:
        print(f"Timing | {label}: {(perf_counter() - started) * 1000:.0f} ms")
