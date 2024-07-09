import os 
import dotenv
import json
import requests
import uuid
import streamlit as st

from datetime import datetime
from typing import Annotated
from typing import Literal
from typing_extensions import TypedDict
from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages

dotenv.load_dotenv()

# 1. setup observability and env vars
os.environ['LANGCHAIN_PROJECT'] = "Sweep chatbot"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# 2. define tool node and tools

# tool error handling
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }
    
    
def create_tool_node_with_fallback(tools:list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )                                           

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
        return {"error": f"{response.status_code} bad request"}
        
tools = [check_calendar]

# 3. define state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    

# 4. assistant node
class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            user_id = configuration.get("user_id", None)
            state = {**state, "user_id": user_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": [result]}
            
    
    
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=1)


#5. bind tools node to llm
primary_assistant_prompt = hub.pull("customer_support_chatbot").partial(time=datetime.now())

part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(tools)


# 6. define graph workflow
builder = StateGraph(State)

# Define nodes: these do the work
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))


# Define edges: these determine how the control flow moves
builder.set_entry_point("assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state

# this is a complete memory for the entire graph.
memory = SqliteSaver.from_conn_string(":memory:")
part_1_graph = builder.compile(checkpointer=memory)
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "user_id": "21458856",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
} 


# 7. streamlit interface
st.set_page_config(page_title="Sweep 🏠",initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
        div#root{
            font-family: Arial, sans-serif;
            background: white;
            color: black;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏠 Sweep Home Services")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to handle user input
def handle_input():
    user_input = st.session_state.input_text
    st.session_state.input_text = ""
    if user_input.lower() in ["quit", "exit", "q"]:
        st.session_state.messages.append({"role": "assistant", "content": "Goodbye!"})
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        for event in part_1_graph.stream({"messages": ("user", user_input)}, config):
            for value in event.values():
                st.session_state.messages.append({"role": "assistant", "content": value["messages"][0].content})

# User input text box
st.text_input("How can we help 👨‍🔧? ", key="input_text", on_change=handle_input)

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**User:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")

