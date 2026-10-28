"""Small public-policy execution records for honest cache-hit provenance.

Never persist questions, model text, arguments, source passages, request IDs,
or customer-tool data in the historical trace.
"""

from datetime import datetime, timezone

FIELDS = {
    "safety": {"action"},
    "cache": {"kind", "hit"},
    "route": {"classifier", "scope", "department"},
    "model_route": {"classifier", "model_tier", "model"},
    "plan": {"calls"},
    "tool_start": {"tool", "call_id", "depends_on"},
    "tool_end": {"tool", "call_id", "ok", "chunk_count"},
    "verify": {"stage", "passed"},
    "synth_start": set(),
}


def public_execution_events(events):
    output = []
    for item in events[:64]:
        if not isinstance(item, dict):
            continue
        name = item.get("event")
        if name not in FIELDS or item.get("kind") == "write":
            continue
        if name in {"tool_start", "tool_end"} and item.get("tool") != "retrieve_policy":
            continue
        if name == "route" and item.get("scope") != "public_information":
            continue
        event = {"event": name, "ms": item.get("ms", 0)}
        for key in FIELDS[name]:
            if key in item:
                event[key] = item[key]
        if name == "plan":
            event["calls"] = [
                {"tool": "retrieve_policy", "depends_on": call.get("depends_on", [])}
                for call in item.get("calls", [])
                if isinstance(call, dict) and call.get("tool") == "retrieve_policy"
            ]
        output.append(event)
    return output


def record_execution(events):
    public = public_execution_events(events)
    return {
        "version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "latency_ms": public[-1]["ms"] if public else 0,
        "events": public,
    }


def read_execution(payload):
    original = payload.get("original_execution")
    if not isinstance(original, dict) or original.get("version") != 1:
        return None
    if not isinstance(original.get("events"), list):
        return None
    return {
        "version": 1,
        "recorded_at": original.get("recorded_at"),
        "latency_ms": original.get("latency_ms", 0),
        "events": public_execution_events(original["events"]),
    }
