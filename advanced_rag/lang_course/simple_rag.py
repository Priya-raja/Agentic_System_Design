import os
import sys
import types
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv(override=True)

# Shim: inject ChatVertexAI before ragas loads
if "langchain_community.chat_models.vertexai" not in sys.modules:
    from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")
    _shim.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _shim

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.llms import llm_factory
from openai import OpenAI

# ---------------------------------------------------------------------------
# 1. LOAD & CHUNK
# ---------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.normpath(os.path.join(script_dir, "..", "data", "Understanding_Climate_Change.pdf"))

loader = PyPDFLoader(path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Loaded {len(documents)} pages → split into {len(chunks)} chunks")

# ---------------------------------------------------------------------------
# 2. EMBED & STORE
# ---------------------------------------------------------------------------

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embedding_model)
print(f"FAISS vectorstore created with {vectorstore.index.ntotal} vectors")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------------------------------------------------------
# 3. RAG CHAIN
# ---------------------------------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]: {doc.page_content}"
        for doc in docs
    )

prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
""")

llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# ---------------------------------------------------------------------------
# 4. TEST QUESTIONS + GROUND TRUTHS
# ---------------------------------------------------------------------------

test_cases = [
    {
        "question": "What are the main causes of climate change?",
        "reference": "The main causes of climate change are burning of fossil fuels, deforestation, and industrial activities that release greenhouse gases."
    },
    {
        "question": "What are the effects of climate change on ecosystems?",
        "reference": "Climate change affects ecosystems by shifting habitat ranges, changing species distributions, causing biodiversity loss, and disrupting marine ecosystems through ocean acidification."
    },
    {
        "question": "What solutions exist to address climate change?",
        "reference": "Solutions include renewable energy, carbon pricing, emissions regulations, carbon capture and storage, and sustainable agriculture."
    },
    {
        "question": "How has Earth's climate changed historically?",
        "reference": "Earth's climate has cycled through glacial and interglacial periods over 650,000 years, with the modern era beginning after the last ice age 11,700 years ago."
    },
    {
        "question": "What is the role of deforestation in climate change?",
        "reference": "Deforestation contributes to climate change by releasing stored carbon and reducing the number of trees available to absorb CO2."
    }
]

# ---------------------------------------------------------------------------
# 5. COLLECT RAG OUTPUTS
# ---------------------------------------------------------------------------

print("\nRunning RAG chain on test cases...")
samples = []
for tc in test_cases:
    retrieved_docs = retriever.invoke(tc["question"])
    contexts = [doc.page_content for doc in retrieved_docs]
    answer = rag_chain.invoke(tc["question"])

    samples.append(SingleTurnSample(
        user_input=tc["question"],
        response=answer,
        retrieved_contexts=contexts,
        reference=tc["reference"]
    ))

    print(f"\nQ: {tc['question']}")
    print(f"A: {answer}")
    print(f"Contexts retrieved: {len(contexts)}")
    print("-" * 60)

# ---------------------------------------------------------------------------
# 6. RAGAS EVALUATION
# ---------------------------------------------------------------------------

print("\nRunning RAGAS evaluation...")
openai_client = OpenAI()
evaluator_llm = llm_factory("gpt-4o", client=openai_client)
lc_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

eval_dataset = EvaluationDataset(samples=samples)

results = evaluate(
    dataset=eval_dataset,
    metrics=[
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        Faithfulness(),
        AnswerRelevancy(embeddings=lc_embeddings),
    ],
    llm=evaluator_llm,
)

# ---------------------------------------------------------------------------
# 7. SCORECARD
# ---------------------------------------------------------------------------

df = results.to_pandas()
metric_cols = [
    "llm_context_precision_with_reference",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
]

print("\nRAGAS SCORECARD — Simple RAG on climate_change.pdf")
print("\nPer-question breakdown:")
for _, row in df.iterrows():
    print(f"\n  Q: {row['user_input'][:65]}")
    for col in metric_cols:
        if col in row:
            val = row[col]
            icon = "✅" if val >= 0.7 else ("⚠️ " if val >= 0.5 else "❌")
            print(f"    {icon} {col:<45}: {val:.3f}")

print("\nAggregate averages:")
for col in metric_cols:
    if col in df.columns:
        avg = df[col].mean()
        icon = "✅" if avg >= 0.7 else ("⚠️ " if avg >= 0.5 else "❌")
        print(f"  {icon} {col:<45}: {avg:.3f}")