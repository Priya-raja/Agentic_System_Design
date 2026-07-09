import os
import sys
import types
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv

if "langchain_community.chat_models.vertexai" not in sys.modules:
    from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")
    _shim.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _shim

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List

load_dotenv(override=True)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = init_chat_model(model="gpt-4o-mini", temperature=0.2)

KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

def create_kb():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = Document(page_content=KNOWLEDGE_BASE, metadata={"source": "knowledge_base.md"})
    chunks = splitter.split_documents([docs])

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory="chroma_kb"
    )
    return vector_store


def demo_basic_rag():
    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    prompt_template = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}

Answer concisely. If you don't know, say "I don't know."
""")

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
    ]

    print("\n--- Basic RAG Chain Test ---\n")
    for q in questions:
        answer = rag_chain.invoke(q)  # ← plain string, not dict
        print(f"Q: {q}")
        print(f"A: {answer}\n")


def demo_rag_with_sources():

    vectorstore = create_kb()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based on the context below. Include which sources you used.

Context:
{context}

Question: {question}

Answer (include sources):"""
    )

    def format_docs_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG with Sources:\n")
    answer = rag_chain.invoke("What are the core components of LangChain?")
    print(f"Q: What are the core components?\n")
    print(f"A: {answer}")

def demo_rag_with_fallback():

    vectorstore = create_kb()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based ONLY on the following context.
If the answer is not in the context, respond with: "I don't have information about that in my knowledge base."

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG with Fallback:\n")

    questions = [
        "What is the pricing for LangSmith?",  # In knowledge base
        "What is the stock price of OpenAI?",  # Not in knowledge base
        "How do I deploy LangChain to AWS?",  # Not in knowledge base
    ]

    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


def demo_structured_rag():
    """RAG with structured output."""

    vectorstore = create_kb()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    class RAGResponse(BaseModel):
        """Structured RAG response."""

        answer: str = Field(description="The answer to the question")
        confidence: str = Field(description="high, medium, or low")
        sources_used: List[str] = Field(description="List of sources referenced")
        follow_up: str = Field(description="Suggested follow-up question")

    structured_llm = llm.with_structured_output(RAGResponse)

    prompt = ChatPromptTemplate.from_template(
        """
Based on the context below, answer the question.

Context:
{context}

Question: {question}

Provide a structured response."""
    )

    def format_docs(docs):
        return "\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}]: {doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | structured_llm
    )
    print("Structured RAG Demo:\n")
    result = rag_chain.invoke("What is LangGraph?")

    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {result.sources_used}")
    print(f"Follow-up: {result.follow_up}")

if __name__ == "__main__":
    # demo_basic_rag()
    # demo_rag_with_sources()
    # demo_rag_with_fallback()
    demo_structured_rag()