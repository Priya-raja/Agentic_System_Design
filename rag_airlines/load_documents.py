"""Compatibility entry point for context.ingest.load_documents."""

from context.ingest.load_documents import *  # noqa: F401,F403


if __name__ == "__main__":
    from runpy import run_module

    run_module("context.ingest.load_documents", run_name="__main__")
