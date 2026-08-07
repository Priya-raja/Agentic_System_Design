from dotenv import load_dotenv
from mem0 import Memory
import json
import os
import json

from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

load_dotenv()

client = OpenAI()

config = {
    "version":"v1.1",
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": OPENAI_API_KEY,
            "model": "text-embedding-3-small"
        }
    },
    "llm":{
        "provider":"openai",
        "config":{"api_key": OPENAI_API_KEY, "model":"gpt-4.1"}
    },
    "vector_store" :{
        "provider":"qdrant",
        "config":{
            "host":"localhost",
            "port":6333
        }
    }
}

memory_client = Memory.from_config(config)

while True:


    user_query = input("> ")

    search_memory = memory_client.search(query = user_query, filters = {"user_id":"priya"})

    memories = [
        f"ID : {mem.get("id")}\nMemory: {mem.get("memory")}" for mem in search_memory.get("results")
    ]

    # print("Found Memories", memories)

    SYSTEM_PROMPT = f"""
                 Here is the context about the user:
                 {json.dumps(memories)}
                """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role":"system", "content": SYSTEM_PROMPT},
            {"role":"user", "content":user_query}
            ]
    )

    ai_response = response.choices[0].message.content
    print("AI:",ai_response)

   # Hey memory this is what ai gave me

    memory_client.add(
        user_id="priya",
        messages = [
            {"role":"user", "content":user_query},
            {"role":"assisstant", "content":ai_response}
        ]
    )
    print("Memory has been saved")