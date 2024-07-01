import os 
import dotenv


from typing_extensions import TypedDict
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


dotenv.load_dotenv()

# 1. setup observability and env vars
os.environ['LANGCHAIN_PROJECT'] = "Sweep chatbot"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2. define tools
# requests_post = load_tools(["requests_post"], allow_dangerous_tools=True)
# tools = [requests_post]
# tool_node = ToolNode(tools)


# 3. graph state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    
graph_builder = StateGraph(State)

# 4. define chatbot node and workflow
llm = ChatOpenAI(temperature=0.9, model_name="gpt-4-turbo")


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

graph_builder.set_entry_point("chatbot")

graph_builder.set_finish_point("chatbot")

graph = graph_builder.compile()

while True:
    user_input = input("user:")
    if user_input in ["exit", "quit", "q"]:
        print("exiting...")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("assistant:", value["messages"][-1].content)