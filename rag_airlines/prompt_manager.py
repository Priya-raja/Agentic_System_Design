import hashlib
from dataclasses import dataclass

from config import (
    ANSWER_PROMPT_ID,
    ANSWER_PROMPT_VERSION,
    PROMPT_ROOT,
)


@dataclass(frozen=True)
class PromptVersion:
    prompt_id: str
    version: str
    content: str
    content_hash: str


def load_answer_prompt(
    version: str | None = None,
) -> PromptVersion:
    selected_version = (
        version or ANSWER_PROMPT_VERSION
    )

    prompt_path = (
        PROMPT_ROOT
        / f"aeronova_answer_{selected_version}.txt"
    )

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt version does not exist: "
            f"{prompt_path}"
        )

    content = prompt_path.read_text(
        encoding="utf-8"
    ).strip()

    if not content:
        raise ValueError(
            f"Prompt file is empty: {prompt_path}"
        )

    content_hash = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    return PromptVersion(
        prompt_id=ANSWER_PROMPT_ID,
        version=selected_version,
        content=content,
        content_hash=content_hash,
    )