"""Read-only metadata lookup for source provenance questions."""

from pydantic import BaseModel, ConfigDict, Field

from agent.tools.base import Tool


class MetadataArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=2000)


def MetadataTool(chunks):
    def inspect(query):
        # Return one record per source file, without exposing or searching policy
        # text. The caller can then cite the source metadata during synthesis.
        documents = {}
        for chunk in chunks:
            metadata = chunk.metadata
            document_id = metadata.get("document_id")
            if not document_id or document_id in documents:
                continue
            is_eu_law = document_id == "POL-EU261-2004"
            documents[document_id] = {
                "document_id": document_id,
                "source_file": metadata.get("source_file"),
                "authority": metadata.get("authority", "unknown"),
                "department": metadata.get("department", "unknown"),
                "status": metadata.get("status", "unknown"),
                "valid_from": metadata.get("valid_from"),
                "valid_to": metadata.get("valid_to"),
                "source_class": "EU law" if is_eu_law else "AeroNova demo corpus",
            }
        selected = sorted(documents.values(), key=lambda item: item["document_id"])
        return {"query": query, "documents": selected}

    return Tool(
        "inspect_metadata",
        MetadataArgs,
        inspect,
        "Inspect loaded corpus document IDs, authority, department and validity dates without retrieving document passages.",
    )
