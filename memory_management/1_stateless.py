"""This is a LLM demo with no memory.IT has no conversation history saved.
Every message is a fresh conversation."""

from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_openai import ChatOpenAI

MODEL = "gpt-4.1-nano"
llm = ChatOpenAI(model= MODEL)

def ask(message: list[dict]) -> str:
    response = llm.invoke(message)
    return response.content

def main() -> None:

    print("Type 'exit' to quit. \n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        #Every call will get only the current message

        message = [
            {"role":"system", "content":"You are a helpful assistant."},
            {"role":"user", "content" : user_input},
        ]
        reply = ask(message)
        print(f"LLM: {reply} \n")


if __name__ == "__main__":
    main()