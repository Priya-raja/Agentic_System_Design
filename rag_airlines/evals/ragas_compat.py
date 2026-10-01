"""
Temporary compatibility fix for RAGAS 0.4.3.

RAGAS imports VertexAI classes from modules removed from
modern langchain-community. AeroNova does not use Vertex AI,
so placeholder classes are sufficient.

Remove this when RAGAS publishes the upstream fix.
"""

import sys
import types

import langchain_community.chat_models
import langchain_community.llms


def apply_ragas_compatibility() -> None:
    vertex_chat_module = (
        "langchain_community.chat_models.vertexai"
    )

    if vertex_chat_module not in sys.modules:
        compatibility_module = types.ModuleType(
            vertex_chat_module
        )

        class ChatVertexAI:
            pass

        setattr(
            compatibility_module,
            "ChatVertexAI",
            ChatVertexAI,
        )

        sys.modules[vertex_chat_module] = (
            compatibility_module
        )

    # RAGAS also imports VertexAI from
    # langchain_community.llms.
    if (
        "VertexAI"
        not in langchain_community.llms.__dict__
    ):

        class VertexAI:
            pass

        setattr(
            langchain_community.llms,
            "VertexAI",
            VertexAI,
        )
