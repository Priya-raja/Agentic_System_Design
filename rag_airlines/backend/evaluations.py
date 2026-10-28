"""Read existing recorded evaluations; never imply these are fresh agent results."""

import json
from config import ROOT_DIR


def dashboard(directory=None):
    directory = directory or ROOT_DIR / "evals/results"
    summaries = sorted(directory.glob("ragas_summary_*.json"))
    if not summaries:
        return {"summary": None, "cases": [], "notice": "No evaluation run available."}
    source = summaries[-1]
    stamp = source.stem.removeprefix("ragas_summary_")
    details = directory / f"ragas_run_{stamp}.jsonl"
    records = (
        [json.loads(line) for line in details.read_text().splitlines() if line.strip()]
        if details.exists()
        else []
    )
    timings = sorted(
        float(r["total_latency_ms"]) for r in records if isinstance(r.get("total_latency_ms"), (float, int))
    )

    def percentile(q):
        if not timings:
            return None
        from math import ceil

        return timings[max(0, ceil(q * len(timings)) - 1)]

    cases = [
        {
            key: r.get(key)
            for key in [
                "case_id",
                "user_input",
                "reference",
                "response",
                "actual_abstention",
                "abstention_correct",
                "citation_errors",
                "measurement_errors",
                "metric_errors",
                "metrics",
                "total_latency_ms",
            ]
        }
        for r in records
    ]
    summary = json.loads(source.read_text())
    current = summary.get("pipeline") == "bounded-agent-v1"
    return {
        "summary": summary,
        "cases": cases,
        "source": source.name,
        "latency_label": "agent request"
        if summary.get("latency_definition") == "agent_only"
        else "evaluation incl. judging",
        "latency_p50_ms": percentile(0.5),
        "latency_p95_ms": percentile(0.95),
        "notice": "Bounded-agent evaluation sample; inspect the run size before comparing quality."
        if current
        else "Historical evaluation of the earlier pipeline. Not a new score for the bounded agent.",
    }
