import os
from pathlib import Path
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).parent.resolve()
# Read local configuration before startup consumers inspect tracing or demo flags.
# Explicit process environment always takes precedence.
load_dotenv(ROOT_DIR / ".env", override=False)

# Corpus
CORPUS_ROOT = ROOT_DIR / "data"

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
# QDRANT_API_KEY is loaded from the environment; never store it in source.
QDRANT_COLLECTION = "aeronova_public_knowledge"
# https://45e450e3-7cc9-44a4-a446-f378fe7e4b62.eu-central-1-0.aws.cloud.qdrant.io

# Embeddings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_SIZE = 1536

# Answer generation
ANSWER_MODEL = "gpt-4.1-mini"
ANSWER_TEMPERATURE = 0

# Prompt version
ANSWER_PROMPT_ID = "aeronova-answer"
ANSWER_PROMPT_VERSION = "v6"
PROMPT_ROOT = ROOT_DIR / "prompts"

# Retrieval
DENSE_TOP_K = 10
BM25_TOP_K = 10
FINAL_TOP_K = 5

DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4
PRIMARY_AUTHORITY_BOOST = 0.002

# Tracing
PHOENIX_ENABLED = os.getenv("PHOENIX_ENABLED", "true").lower() in {"1", "true", "yes"}
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "aeronova-rag")
PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
PHOENIX_PUBLIC_URL = os.getenv("PHOENIX_PUBLIC_URL", "http://localhost:6006")

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

# Keep an answer for 24 hours.
ANSWER_CACHE_TTL_SECONDS = 24 * 60 * 60

# Briefly cache exact questions with no matching policy context.
NO_CONTEXT_CACHE_TTL_SECONDS = 300

# Increment when routing, validation or generation logic changes.
ANSWER_PIPELINE_VERSION = "classic-rag-v1-intent-routing"

# Conservative semantic reuse within matching policy and prompt scopes.
SEMANTIC_CACHE_THRESHOLD = 0.97
SEMANTIC_CACHE_MAX_ENTRIES = 100

# Public RAG departments supported by the current corpus.
ACTIVE_RAG_DEPARTMENTS = {
    "baggage",
    "disruptions",
    "refunds_claims",
    "cabin_onboard",
    "general_policies",
    "upgrades",
    "rewards_loyalty",
}
INDEX_SCHEMA_VERSION = "departments-v5-routes"

# Policy authority requirements are explicit; secondary menu guides are acceptable.
REQUIRE_PRIMARY = {department: department != "cabin_onboard" for department in ACTIVE_RAG_DEPARTMENTS}
