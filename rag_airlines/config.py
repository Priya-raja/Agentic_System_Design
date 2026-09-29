from pathlib import Path


ROOT_DIR = Path(__file__).parent.resolve()

# Corpus
CORPUS_ROOT = ROOT_DIR / "data"

# Qdrant
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "aeronova_public_knowledge"

# Embeddings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_SIZE = 1536

# Answer generation
ANSWER_MODEL = "gpt-4.1-mini"
ANSWER_TEMPERATURE = 0

# Prompt version
ANSWER_PROMPT_ID = "aeronova-answer"
ANSWER_PROMPT_VERSION = "v4"
PROMPT_ROOT = ROOT_DIR / "prompts"

# Retrieval
DENSE_TOP_K = 10
BM25_TOP_K = 10
FINAL_TOP_K = 5

DENSE_WEIGHT = 0.6
BM25_WEIGHT = 0.4
PRIMARY_AUTHORITY_BOOST = 0.002

# Tracing
PHOENIX_ENABLED = True
PHOENIX_PROJECT_NAME = "aeronova-rag"
PHOENIX_ENDPOINT = (
    "http://localhost:6006/v1/traces"
)

# JEV
JEV_API_URL = "https://opencode.ai/zen/v1/systemone"
JEV_MODEL = "jev-1.13-free"

JEV_TOPIC_MIN_CONFIDENCE = 0.65
JEV_ROUTE_MIN_CONFIDENCE = 0.70
JEV_TIMEOUT_SECONDS = 15

CHEAP_ANSWER_MODEL = "gpt-4.1-nano"
