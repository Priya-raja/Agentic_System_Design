"""Typed tool registration. The registry is an allowlist, never dynamic imports."""

from dataclasses import dataclass, field
from typing import Any, Callable
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    tool: str
    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    call_id: int = -1
    private: bool = False

    @classmethod
    def failure(cls, tool, error, call_id=-1):
        return cls(tool=tool, ok=False, error=error, call_id=call_id)


@dataclass
class Tool:
    name: str
    schema: type[BaseModel]
    handler: Callable[..., dict[str, Any]]
    description: str = ""
    requires_auth: bool = False
    requires_confirmation: bool = False
    timeout: float = 25
    customer_id: str | None = None

    def run(self, **kwargs) -> ToolResult:
        args = self.schema.model_validate(kwargs)
        data = self.handler(**args.model_dump())
        return ToolResult(tool=self.name, ok=True, data=data, private=self.requires_auth)


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool):
        if tool.name in self.tools:
            raise ValueError("Duplicate tool")
        self.tools[tool.name] = tool
        return self

    def get(self, name):
        return self.tools[name]

    def descriptions(self):
        return [
            {
                "name": t.name,
                "description": t.description,
                "requires_auth": t.requires_auth,
                "requires_confirmation": t.requires_confirmation,
                "schema": t.schema.model_json_schema(),
            }
            for t in self.tools.values()
        ]
