import chromadb

client = chromadb.Client()

collection_name = "test_collection"
collection = client.get_or_create_collection(collection_name)


documents = [
    {"id": "doc1", "text":"Hello, world!"},
    {"id":"doc2", "text":'Hows the day going?'},
    {"id":"doc3", "text":'I am good, thank you!'},
]

for doc in documents:
    collection.upsert(
    documents=[
       doc["text"]
    ],
    ids=[doc["id"]]
)

query_text ="Hello World"

results = collection.query(
    query_texts=[query_text], # Chroma will embed this for you
    n_results=2 # how many results to return
)

print(results)