"""Bounded Redis semantic cache for conservative public-policy paraphrases."""

import hashlib
import json
import math
import re
import time
from datetime import date

from config import (
    ACTIVE_RAG_DEPARTMENTS,
    ANSWER_CACHE_TTL_SECONDS,
    ANSWER_PIPELINE_VERSION,
    EMBEDDING_MODEL,
    SEMANTIC_CACHE_MAX_ENTRIES,
    SEMANTIC_CACHE_THRESHOLD,
)
from context.memory.answer_cache import normalize_question, redis_client
from safety.heuristics import assess_user_input
from safety.policy import cache_payload_allowed

# Preserve meaningful words (including routes, fares, dates, quantities, and
# eligibility terms). Only known equivalent words and question boilerplate vary.
ALIASES = {"bags": "baggage", "bag": "baggage", "luggage": "baggage",
           "meals": "meal", "seats": "seat", "kilograms": "kg",
           "includes": "include", "included": "include", "permitted": "allowed"}
BOILERPLATE = set("what whats is are the a an how much many please tell me about can could you explain do does of for on in to from".split())
UNSAFE = re.compile(
    r"\b(not|no|except|exception|compare|versus|vs|difference|calculate|total|"
    r"my|our|booking|reservation|passport|refund|compensation|eligible|eligibility|"
    r"today|tomorrow|yesterday|next|last|current|latest)\b", re.I,
)


def semantic_cache_allowed(question: str, topic: str | None) -> bool:
    """Require a known metadata topic and avoid conditional/personal queries."""
    return bool(
        assess_user_input(question).cache_allowed
        and topic in ACTIVE_RAG_DEPARTMENTS
        and not UNSAFE.search(question)
        and not re.search(r"\b(if|unless|and|or)\b", question, re.I)
    )


def question_signature(question: str) -> list[str]:
    tokens = re.findall(r"[\w]+(?:[.-][\w]+)*", normalize_question(question))
    return [ALIASES.get(token, token) for token in tokens if token not in BOILERPLATE]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    denominator = math.sqrt(sum(x*x for x in left) * sum(x*x for x in right))
    return sum(x*y for x, y in zip(left, right)) / denominator if denominator else 0.0


class SemanticAnswerCache:
    """Scan a small, expiring bucket of validated answers in the same scope."""

    def __init__(self, embeddings=None, client=None):
        self.embeddings = embeddings
        self.client = client if client is not None else redis_client

    def _embed(self, question: str) -> list[float]:
        if self.embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        return self.embeddings.embed_query(normalize_question(question))

    @staticmethod
    def _key(metadata: dict) -> str:
        scope = {**metadata, "pipeline_version": ANSWER_PIPELINE_VERSION,
                 "embedding_model": EMBEDDING_MODEL, "as_of": date.today().isoformat()}
        digest = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
        return f"aeronova:semantic-answer:{digest}"

    def lookup(self, *, question: str, topic: str, policy_year: int | None,
               prompt_version: str, corpus_version: str, prompt_hash: str) -> dict | None:
        if not assess_user_input(question).cache_allowed:
            return None
        metadata = dict(topic=topic, policy_year=policy_year, prompt_version=prompt_version,
                        corpus_version=corpus_version, prompt_hash=prompt_hash)
        if corpus_version == "missing-index-state":
            return None
        try:
            entries = self.client.lrange(self._key(metadata), 0, SEMANTIC_CACHE_MAX_ENTRIES - 1)
            candidates = []
            for value in entries:
                entry = json.loads(value)
                if (cache_payload_allowed(entry['payload'])
                        and time.time() - entry['created_at'] < ANSWER_CACHE_TTL_SECONDS
                        and entry['signature'] == question_signature(question)):
                    candidates.append(entry)
            if not candidates:
                return None
            embedding = self._embed(question)
            best = max(candidates, key=lambda entry: cosine_similarity(embedding, entry['embedding']))
            if cosine_similarity(embedding, best['embedding']) >= SEMANTIC_CACHE_THRESHOLD:
                return best['payload']
        except Exception as error:
            # Embedding and Redis failures must not stop the normal RAG pipeline.
            print(f"Semantic cache unavailable ({type(error).__name__})")
        return None

    def store(self, *, question: str, answer: dict, metadata: dict) -> None:
        if not assess_user_input(question).cache_allowed:
            return
        if not cache_payload_allowed(answer):
            return
        # Cache grounded answers only; a refusal for one question need not apply
        # to a paraphrase. Preserve retrieval payloads so citations still display.
        if not answer.get('response', {}).get('answerable'):
            return
        if metadata.get('corpus_version') == 'missing-index-state':
            return
        try:
            entry = dict(created_at=time.time(), signature=question_signature(question),
                         embedding=self._embed(question), payload=answer)
            key = self._key(metadata)
            with self.client.pipeline() as pipeline:
                pipeline.lpush(key, json.dumps(entry, allow_nan=False))
                pipeline.ltrim(key, 0, SEMANTIC_CACHE_MAX_ENTRIES - 1)
                pipeline.expire(key, ANSWER_CACHE_TTL_SECONDS)
                pipeline.execute()
        except Exception as error:
            print(f"Could not cache semantic answer ({type(error).__name__})")
