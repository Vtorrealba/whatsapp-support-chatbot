import os 
import dotenv
import json

from typing import Annotated
from typing import Literal
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.messages import ToolMessage
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

# 2. define tools and tool node
tool = load_tools(["requests_post"], allow_dangerous_tools=True)
tools = [tool]
class BasicToolNode:
    def __init__(self, tools:list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
        
    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No messages found in inputs")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content= json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

tool_node = BasicToolNode(tools=[tools])

# 3. graph state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    
graph_builder = StateGraph(State)

# 4. define chatbot node and workflow
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tool_node)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("chatbot")

graph_builder.set_finish_point("chatbot")

graph = graph_builder.compile()

