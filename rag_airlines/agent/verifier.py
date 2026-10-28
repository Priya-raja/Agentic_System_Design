"""Deterministic gates before and after synthesis; no model-based grading."""

from opentelemetry import trace

tracer = trace.get_tracer(__name__)


import json
from dataclasses import dataclass
from langchain_core.documents import Document
from agent.answers import validate_citations, validate_measurements, validate_format
from safety.policy import ensure_safe_output, filter_safe_context, SafetyViolation
from config import REQUIRE_PRIMARY


@dataclass(frozen=True)
class VerificationReport:
    passed: bool
    citation_errors: tuple[str, ...] = ()
    measurement_errors: tuple[str, ...] = ()
    format_errors: tuple[str, ...] = ()
    safety_error: str | None = None

    @property
    def errors(self):
        return (
            self.citation_errors
            + self.measurement_errors
            + self.format_errors
            + ((self.safety_error,) if self.safety_error else ())
        )


class Verifier:
    def __init__(self, retrieval_results=None, tool_results=None):
        self.results = list(retrieval_results or [])
        self.tools = list(tool_results or [])
        known = {d.metadata.get("chunk_id") for d, _ in self.results}
        for result in self.tools:
            if not result.ok or result.private:
                continue
            if result.tool == "retrieve_policy":
                for chunk in result.data.get("chunks", []):
                    if chunk["metadata"].get("chunk_id") not in known:
                        self.results.append(
                            (
                                Document(page_content=chunk["content"], metadata=chunk["metadata"]),
                                chunk["score"],
                            )
                        )
                        known.add(chunk["metadata"].get("chunk_id"))
            elif result.tool == "inspect_metadata":
                for item in result.data.get("documents", []):
                    chunk_id = f"META-{item['document_id']}"
                    if chunk_id in known:
                        continue
                    self.results.append((
                        Document(
                            page_content=json.dumps(item, sort_keys=True),
                            metadata={
                                "chunk_id": chunk_id,
                                "document_id": item["document_id"],
                                "source_file": item.get("source_file", "corpus metadata"),
                                "authority": item.get("authority", "unknown"),
                                "department": item.get("department", "unknown"),
                                "status": item.get("status"),
                                "valid_from": item.get("valid_from"),
                                "valid_to": item.get("valid_to"),
                                "source_class": item.get("source_class"),
                            },
                        ),
                        1.0,
                    ))
                    known.add(chunk_id)
            else:
                self.results.append(
                    (
                        Document(
                            page_content=json.dumps(result.data, sort_keys=True, default=str),
                            metadata={
                                "chunk_id": f"TOOL-{result.call_id}",
                                "document_id": result.tool,
                                "source_file": "verified tool result",
                                "authority": "primary",
                                "department": "flight_status",
                                "tool_call_id": result.call_id,
                            },
                        ),
                        1.0,
                    )
                )

    @tracer.start_as_current_span("aeronova.verify.evidence", record_exception=False, set_status_on_exception=False)
    def evidence_errors(self):
        errors = []
        policy_calls = [result for result in self.tools if result.tool == "retrieve_policy"]
        if len(policy_calls) > 1 and any(not result.data.get("chunks") for result in policy_calls):
            errors.append("A requested department has no supporting context")
        if len(filter_safe_context(self.results)) != len(self.results):
            errors.append("Unsafe context")
        metadata_only = any(result.tool == "inspect_metadata" for result in self.tools)
        departments = set() if metadata_only else {d.metadata.get("department") for d, _ in self.results}
        for department in departments:
            if REQUIRE_PRIMARY.get(department, True) and not any(
                d.metadata.get("authority") == "primary"
                for d, _ in self.results
                if d.metadata.get("department") == department
            ):
                errors.append("Primary evidence required for " + str(department))
        trace.get_current_span().set_attribute("verification.passed", not errors)
        trace.get_current_span().set_attribute("verification.error_count", len(errors))
        return errors

    @tracer.start_as_current_span("aeronova.verify", record_exception=False, set_status_on_exception=False)
    def check(self, response):
        citations = validate_citations(response, self.results)
        if response.answerable:
            citations += self.evidence_errors()
        measurements = validate_measurements(response, self.results) if response.answerable else []
        formatting = validate_format(response)
        error = None
        try:
            ensure_safe_output(response, self.results)
        except SafetyViolation:
            error = "Response or evidence failed safety verification"
        trace.get_current_span().set_attribute("verification.passed", not citations and not measurements and not formatting and error is None)
        trace.get_current_span().set_attribute("verification.citation_errors", len(citations))
        trace.get_current_span().set_attribute("verification.measurement_errors", len(measurements))
        trace.get_current_span().set_attribute("verification.format_errors", len(formatting))
        return VerificationReport(
            not citations and not measurements and not formatting and error is None, tuple(citations), tuple(measurements), tuple(formatting), error
        )
