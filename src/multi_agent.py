import os
import dotenv
import requests
import functools
import operator
import uuid
from datetime import datetime
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field, EmailStr
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import ToolMessage, HumanMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool, BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, START
from typing import Annotated, Type, Optional, Sequence, TypedDict
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils.agent_helpers import _prompt_text_loader

# 1. setup observability and env vars
dotenv.load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = "Sweep chatbot"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2. define tool node and tools, define the schemas for the tool inputs
class CheckCalendar(BaseModel):
    date: str = Field(..., description="The date to check availability in the calendar. input should be a future date in MM/DD/YYYY format (e.g., 11/22/2024)")

class AppointmentBooking(BaseModel):
    name: str = Field(description="The name of the person or company booking the appointment", min_length=2, max_length=100)
    email: EmailStr = Field(..., description="The email of the person or company booking the appointment")
    date: str = Field(..., description="The date and time of the appointment in ISO 8601 format (e.g., 2024-07-27T13:00:00.000Z)")

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
    """
    Check the availability of a date provided by the user in the calendar.
    convert the date provided by the user to the mm/dd/yyyy format
    """
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
check_calendar.description = _prompt_text_loader(check_calendar_prompt)

book_appointment_prompt = hub.pull("book_appointment_tool")
book_appointment_tool_description = _prompt_text_loader(book_appointment_prompt)

class BookAppointmentTool(BaseTool):
    name = "book_appointment"
    description = book_appointment_tool_description
    args_schema: Type[BaseModel] = AppointmentBooking

    def _run(
        self, name: str, email: str, date: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Book an appointment."""
        webhook_url = os.getenv("PCP_WEBHOOK_URL")
        payload = {
            "name": name,
            "email": email,
            "date": date,
        }
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            return "Appointment successfully booked."
        else:
            return f"Failed to book appointment. Status code: {response.status_code}"

    async def _arun(
        self, name: str, email: str, date: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Asynchronously book an appointment."""
        # Implement async version if needed
        raise NotImplementedError("Async version not implemented")

    def __call__(self, name: str, email: str, date: str) -> str:
        return self._run(name, email, date)

# Create an instance of the tool
book_appointment = BookAppointmentTool()
scheduling_tools = [check_calendar, book_appointment]

# define helper that facilitates the creation of the agent
def create_agent(llm:ChatOpenAI, tools: list, system_prompt:str) -> AgentExecutor:
    prompt = hub.pull(system_prompt)
    if system_prompt == "sweep_scheduling":
        prompt = hub.pull(system_prompt).partial(time=datetime.now())

    agent = create_openai_tools_agent(tools=tools, llm=llm, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

members = ["Briefer", "Scheduler"]
# the supervisor is another LLM node.
options = members
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}

prompt = hub.pull("sweep_agent_routing").partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI(model="gpt-4-1106-preview")

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

# 4. define your agents

# AGENT 1: the scheduling agent

scheduling_agent = create_agent(llm, scheduling_tools, "sweep_scheduling")
scheduling_node = functools.partial(agent_node, agent=scheduling_agent, name="Scheduler")

# AGENT 2: the briefing agent
briefing_agent = create_agent(llm, [check_calendar], "sweep_briefing")
briefing_node = functools.partial(agent_node, agent=briefing_agent, name="Briefer")

# 6. define graph workflow
workflow = StateGraph(AgentState)
workflow.add_node("Scheduler", scheduling_node)
workflow.add_node("Briefer", briefing_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, END)
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.add_edge(START, "supervisor")

# this is a complete memory for the entire graph.
memory = SqliteSaver.from_conn_string(":memory:")

multi_agent_graph = workflow.compile(checkpointer=memory)

