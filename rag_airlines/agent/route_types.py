"""Typed route data used by the deterministic policy router."""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

ToolName = Literal[
    "retrieve_policy", "inspect_metadata", "flight_status", "booking_lookup", "list_claims", "claim_check", "refund_calc"
]


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)


class ToolPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    calls: list[ToolCall] = Field(default_factory=list, max_length=4)
    explanation: str = Field(default="", max_length=300)
    department: str = "general_support"
    request_scope: Literal["public_information", "customer_specific"] = "public_information"
    risk_category: Literal[
        "safe",
        "prompt_injection",
        "privacy_exfiltration",
        "fraud",
        "dangerous_goods_evasion",
        "threatening_harm",
        "uncertain",
    ]

    @model_validator(mode="after")
    def validate_dependencies(self):
        for index, call in enumerate(self.calls):
            if any(dep < 0 or dep >= index for dep in call.depends_on):
                raise ValueError("Dependencies must reference earlier calls")
            if len(set(call.depends_on)) != len(call.depends_on):
                raise ValueError("Duplicate dependency")
        return self
