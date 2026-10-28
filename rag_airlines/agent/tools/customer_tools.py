"""Read-only private tools. Raw records never enter synthesis or public caches."""

import json
from pydantic import BaseModel, ConfigDict, Field
from agent.tools.base import Tool


class BookingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pnr: str = Field(pattern=r"^[A-Z0-9]{5,8}$")


class ClaimArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str = Field(pattern=r"^[A-Z0-9-]{3,40}$")


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def customer_tools(refund_agent):
    def booking_lookup(pnr):
        records = [
            json.loads(line)
            for line in (refund_agent.crm_dir / "bookings.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if any(record.get("synthetic") is not True for record in records):
            raise ValueError("Non-synthetic store")
        record = next(
            (r for r in records if r["pnr"] == pnr and r["customer_id"] == refund_agent.customer_id), None
        )
        if record is None:
            raise ValueError("Booking not found for bound customer")
        return {
            key: record[key]
            for key in [
                "pnr",
                "flight_number",
                "origin",
                "destination",
                "departure_date",
                "cabin",
                "fare_family",
                "status",
            ]
        }

    def refund_calc(claim_id):
        claim = refund_agent.check(claim_id)
        if not claim["can_proceed"]:
            raise ValueError("Approved claim required")
        return {
            **claim,
            "notice": "Calculation only; no payment sent. Use the confirmed refund workflow to proceed.",
        }

    tools = [
        Tool("booking_lookup", BookingArgs, booking_lookup, "Bound customer booking lookup.", True),
        Tool(
            "list_claims",
            EmptyArgs,
            lambda: {"claims": refund_agent.list_claims()},
            "List bound customer claims.",
            True,
        ),
        Tool("claim_check", ClaimArgs, refund_agent.check, "Check a bound customer claim.", True),
        Tool(
            "refund_calc",
            ClaimArgs,
            refund_calc,
            "Read-only approved refund calculation; confirmation required.",
            True,
            True,
        ),
    ]

    for tool in tools:
        tool.customer_id = refund_agent.customer_id
    return tools
