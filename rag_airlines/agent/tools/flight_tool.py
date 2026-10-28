import json
from datetime import date
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from agent.tools.base import Tool
from config import ROOT_DIR


class FlightArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    flight_number: str = Field(pattern=r"^AN\d{3}$")
    departure_date: date


def FlightStatusTool(snapshot_path: Path | None = None):
    path = snapshot_path or ROOT_DIR / "data/operations/flight_status.json"

    def lookup(flight_number, departure_date):
        records = json.loads(path.read_text())
        item = next(
            (
                r
                for r in records
                if r["flight_number"] == flight_number and r["departure_date"] == departure_date.isoformat()
            ),
            None,
        )
        if not item:
            raise ValueError("No archived flight snapshot")
        return {**item, "live": False, "notice": "Archived fictional snapshot, not live flight information."}

    return Tool(
        "flight_status",
        FlightArgs,
        lookup,
        "Read-only archived fictional flight status. Exact flight and ISO date required.",
    )
