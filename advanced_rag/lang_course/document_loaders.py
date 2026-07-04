import os
import sys
import types
import tempfile
import warnings
from pathlib import Path
from langchain_core.documents import Document
warnings.filterwarnings("ignore", category=DeprecationWarning)

if "langchain_community.chat_models.vertexai" not in sys.modules:
    from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")
    _shim.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _shim

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader, PyPDFLoader,DirectoryLoader
from langchain_community.document_loaders import WebBaseLoader
load_dotenv()


def load_text_file():
    file_path = "sample.txt"

    # Write the file first, then close it, then load it
    with open(file_path, "w") as f:
        f.write("Two roads diverged in a yellow wood,\n")
        f.write("Why don't scientists trust atoms? Because they make up everything!\n")

    # File is fully written and closed here — safe to load
    try:
        loader = TextLoader(file_path)
        documents = loader.load()
        print(type(documents))
        print(f"Loaded {len(documents)} documents from {file_path}")
        print(f"Content:\n{documents[0].page_content}")
        print(f"Metadata: {documents[0].metadata}")
        return documents
    except Exception as e:
        print(f"Error loading text file: {e}")
        return None

def web_loader():
        url = "https://en.wikipedia.org/wiki/Web_scraping"
        loader = WebBaseLoader(url, bs_kwargs={"parse_only":None})
        documents = loader.load()
       
        print(f"Loaded {len(documents)} documents from {url}")
        print(f"Source: {documents[0].metadata.get('source', 'N/A')}")
        print(f"Content length: {len(documents[0].page_content)} characters")
        print(f"Preview: {documents[0].page_content[:200]}...")
        return documents

def lazy_loader():
        with tempfile.TemporaryDirectory() as tmpdir:
           
            # Create sample files
            for i in range(5):
                path = Path(tmpdir)/f"doc_{i}.txt"
                path.write_text(f"This is document {i}. It contains my favorite things.")
            loader = DirectoryLoader(tmpdir, glob="*.txt",loader_cls=TextLoader)
        
            for doc in loader.lazy_load():
                print(f"Document {doc.metadata['source']}: {doc.page_content}")
                print("Metadata:", doc.metadata["source"])
                print("-"*50)
            return loader

def doc_structure():
    doc = Document(
        page_content="This is a sample document.",
        metadata={
            "source": "manual_creation.txt",
            "author": "Paulo",
            "length": 30,
            "tags": ["sample", "test"],
            "created_at": "2024-06-01",
        },
    )

    print("Document Structure:")
    print(f"  page_content (type): {type(doc.page_content)}")
    print(f"  page_content: {doc.page_content}")
    print(f"  metadata: {doc.metadata}") 


def pdf_loader(pdf_path:str):
    pdf_loader = PyPDFLoader(pdf_path)
    documents = pdf_loader.load()
    print(f"Loaded {len(documents)} documents from {pdf_path}")
    for i, doc in enumerate(documents):
       
        print(f"  Document {i +1}: Content: {doc.page_content[:100]}")
        print(f"  Metadata: {doc.metadata}")
        print("-"*50)
    return documents

if __name__ == "__main__":
    # load_text_file()
    # web_loader()
    # lazy_loader()
    # doc_structure()
    pdf_loader("../data/langchain_demo.pdf")