import os 
import dotenv
import json
import requests

from datetime import datetime
from typing import Annotated
from typing import Literal
from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph
from langgraph.graph.message import AnyMessage, add_messages

dotenv.load_dotenv()

# 1. setup observability and env vars
os.environ['LANGCHAIN_PROJECT'] = "Sweep chatbot"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2. define tool node and tools
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


    
@tool("check_calendar")
def check_calendar(date: str) -> dict:
    """Check the availability of a date provided by the user in the calendar. 
       the date should be in the format MM/DD/YYYY
    """
    webhook_url = os.getenv("WEBHOOK_URL")
    response = requests.post(webhook_url, json={"start_date": date})    
    if response.status_code == 200:
        response_data = response.json()
        if "slots" not in response_data or "slots" == []:
            return {"error": "No availability found"}
        else:
            return {"availability": response_data["slots"]}
    else:
        return {"error": "{response.status_code} bad request"}
        

tools = [check_calendar]
check_calendar.invoke("7/1/2024")
tool_node = BasicToolNode(tools=[tools])

# 3. graph state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    
graph_builder = StateGraph(State)

# 4. define chatbot node
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=1)
llm_with_tools = llm.bind_tools(tool_node)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 5. define graph workflow
graph_builder.add_node("chatbot", chatbot)

graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("chatbot")

graph_builder.set_finish_point("chatbot")

graph = graph_builder.compile()

