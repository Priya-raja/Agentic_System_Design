from educosys_claude.config import config
from educosys_claude.observability.logger import get_logger

logger = get_logger(__name__)


def get_llm():
    """Return the configuration chat model."""
    provider = config["llm"]["provider"]
    model = config["llm"]["model"]

    logger.info("Using LLM provider: %s, model: %s", provider, model)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model)

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model)


def get_embedder():
    """Return the configuration embedding model."""
    provider = config["embeddings"]["provider"]
    model = config["embeddings"]["model"]

    logger.info("Using embedding provider: %s, model: %s", provider, model)

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model)
    
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model)

    