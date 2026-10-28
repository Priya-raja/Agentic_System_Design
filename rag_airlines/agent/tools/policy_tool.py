from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from agent.tools.base import Tool
from context.indexers.retrieve import hybrid_retrieve
from safety.policy import filter_safe_context


class PolicyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    department: Literal[
        "baggage",
        "disruptions",
        "refunds_claims",
        "rewards_loyalty",
        "upgrades",
        "cabin_onboard",
        "general_policies",
    ]
    query: str = Field(min_length=3, max_length=2000)

    @field_validator("query")
    @classmethod
    def safe_query(cls, value):
        from safety.heuristics import assess_user_input, BLOCK, SECURE, REVIEW

        if assess_user_input(value).action in {BLOCK, SECURE, REVIEW}:
            raise ValueError("Unsafe policy query")
        return value


def RetrievePolicyTool(chunks, vector_store, final_k=5):
    def retrieve(department, query):
        results = hybrid_retrieve(query, chunks, vector_store, department=department, final_k=final_k)
        safe = filter_safe_context(results)
        if results and not safe:
            raise ValueError("Context needs review")
        return {
            "department": department,
            "chunks": [
                {"content": d.page_content, "metadata": d.metadata, "score": float(score)}
                for d, score in safe
            ],
        }

    return Tool(
        "retrieve_policy",
        PolicyArgs,
        retrieve,
        "Retrieve public policy evidence from exactly one department.",
    )
