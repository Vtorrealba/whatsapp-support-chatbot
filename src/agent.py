import os 
import dotenv
import requests

from datetime import datetime
from langchain import hub
from langchain_anthropic import ChatAnthropic
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field, EmailStr
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import tool, BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict
from typing import (
    Annotated,
    Type, 
    Optional, 
)
from utils.agent_helpers import _tool_prompt_loader


# 1. setup observability and env vars
dotenv.load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = "Sweep chatbot"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# 2. define tool node and tools, define the schemas for the tool inputs
class CheckCalendar(BaseModel):
    date: str = Field(..., description="The date to check availability in the calendar. input should be a future date in MM/DD/YYYY format (e.g., 11/22/2024)")
class AppointmentBooking(BaseModel):
    name: str = Field(description="The name of the person or company booking the appointment", min_length=2, max_length=100)
    email: EmailStr = Field(..., description="The email of the person or company booking the appointment")
    phone: str = Field(description="The phone number of the person or company booking the appointment. trim the plus sign if present", pattern=r'^1?\d{9,15}$')
    date: str = Field(..., description="The date and time of the appointment in ISO 8601 format (e.g., 2024-07-27T13:00:00.000Z)")
    reason: str = Field(description="The reason of the appointment booking", max_length=100)

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

@tool("check_calendar", args_schema=CheckCalendar)
def check_calendar(date:str) -> dict:
    """Check the availability of a date provided by the user in the calendar. convert the date provided by the user to the mm/dd/yyyy format""" # don't ever touch this
    webhook_url = os.getenv("CC_WEBHOOK_URL")
    response = requests.post(webhook_url, json={"start_date": date})    
    if response.status_code == 200:
        response_data = response.json()
        if "slots" not in response_data or "slots" == []:
            return {"error": "No availability found"}
        else:
            return {"availability": response_data["slots"]}
    else:
        return {"error": f"{response.status_code} bad request"}

check_calendar_prompt = hub.pull("check_calendar_tool")
check_calendar.description = _tool_prompt_loader(check_calendar_prompt)


book_appointment_prompt = hub.pull("book_appointment_tool")
book_appointment_tool_description = _tool_prompt_loader(book_appointment_prompt)
class BookAppointmentTool(BaseTool):
    name = "book_appointment"
    description = book_appointment_tool_description
    args_schema: Type[BaseModel] = AppointmentBooking

    def _run(
        self, name: str, email: str, phone: str, date: str, reason: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Book an appointment."""
        webhook_url = os.getenv("PCP_WEBHOOK_URL")
        payload = {
            "name": name,
            "email": email,
            "phone": phone,
            "date": date,
            "reason": reason
        }
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            return "Appointment successfully booked."
        else:
            return f"Failed to book appointment. Status code: {response.status_code}"

    async def _arun(
        self, name: str, email: str, phone: str, date: str, reason: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Asynchronously book an appointment."""
        # Implement async version if needed
        raise NotImplementedError("Async version not implemented")

# Create an instance of the tool
book_appointment = BookAppointmentTool()


tools = [check_calendar, book_appointment]


# 3. define state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    

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
        return {"messages": result}
            
    
    
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
thread_id = "1"

config = {
    "configurable": {
        "user_id": "21458856",
        "thread_id": thread_id,
    }
}


while True:
    user_input = input("Client:")
    if user_input.lower() in ["q","quit","exit"]:
        print("assistant: Goodbye!\n")
        break
    else:
        for event in part_1_graph.stream({"messages": ("user", user_input)}, config):
            for value in event.values():
                try:
                    print(f"assistant: {value['messages'].content}\n")
                except:
                    print(f"tool: {value['messages'][0].content}\n")



