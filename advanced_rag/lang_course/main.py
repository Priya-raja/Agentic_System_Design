from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from typing_extensions import Annotated
from langchain_core import __version__ as core_version
# from langgraph import __version__ as lg_version
from langchain_openai import ChatOpenAI



print(f"Langchain Core version: {core_version}")
# print(f"Langgraph version: {lg_version}")

# model = ChatAnthropic(model="claude-haiku-4-5-20251001")

load_dotenv()
def get_weather(
    location: Annotated[str, ..., "Location as city and state."]
) -> str:
    """Get the weather at a location."""
    return "It's sunny."


def main():

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
 

    response = llm.invoke("Say setup complete in one word")
    print(f"response from openai {response}")

    # anthropic_llm_response  = anthropic_llm.invoke("Say setup complete in one word")
    # print(f"response from anthropic {anthropic_llm_response}")
    # print("Test complete!")

    model = ChatAnthropic(model="claude-haiku-4-5-20251001")
    model_with_tools = model.bind_tools([get_weather])
    response1 = model_with_tools.invoke("Which city is hotter today: LA or NY?")
    print(f"response from anthropic with tools {response1.content}")



if __name__ == "__main__":
    main()
