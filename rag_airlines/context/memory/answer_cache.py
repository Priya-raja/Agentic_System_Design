from config import ROOT_DIR
import hashlib
import json
from typing import Any

import redis
from safety.policy import cache_payload_allowed
from redis.exceptions import RedisError
from config import (
    ANSWER_CACHE_TTL_SECONDS,
    ANSWER_PIPELINE_VERSION,
    REDIS_URL,
)


ROOT = ROOT_DIR
INDEX_STATE_FILE = ROOT / "index_state.json"

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=0.5,
    socket_timeout=0.5,
)


def normalize_question(question: str) -> str:
    return " ".join(
        question.lower().strip().split()
    )


def get_index_hash() -> str:
    """
    Rebuilding the index changes index_state.json and therefore
    invalidates answers produced from the previous index.
    """
    if not INDEX_STATE_FILE.exists():
        return "missing-index-state"

    content = INDEX_STATE_FILE.read_bytes()

    return hashlib.sha256(content).hexdigest()[:16]


def create_answer_cache_key(
    question: str,
    prompt_hash: str,
) -> str:
    identity = {
        "question": normalize_question(question),
        "prompt_hash": prompt_hash,
        "index_hash": get_index_hash(),
        "pipeline_version": ANSWER_PIPELINE_VERSION,
    }

    serialized = json.dumps(
        identity,
        sort_keys=True,
    )

    digest = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

    return f"aeronova:answer:{digest}"


def get_cached_answer(
    cache_key: str,
) -> dict[str, Any] | None:
    try:
        value = redis_client.get(cache_key)

        if value is None:
            return None

        payload = json.loads(value)
        if not cache_payload_allowed(payload):
            print("Safety: cached payload requires review.")
            return None
        return payload

    except (RedisError, json.JSONDecodeError) as error:
        print(f"Cache unavailable ({type(error).__name__})")
        return None


def set_cached_answer(
    cache_key: str,
    payload: dict[str, Any],
    *,
    ttl_seconds: int = ANSWER_CACHE_TTL_SECONDS,
) -> None:
    serialized = json.dumps(payload, default=str)
    if not cache_payload_allowed(payload):
        print("Safety: sensitive or unsafe answer payload was not cached.")
        return
    try:
        redis_client.set(
            cache_key,
            serialized,
            ex=ttl_seconds,
        )

    except RedisError as error:
        print(f"Could not cache answer ({type(error).__name__})")
