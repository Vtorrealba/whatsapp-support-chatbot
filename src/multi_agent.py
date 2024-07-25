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
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import tool, BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated, Type, Optional, Sequence, TypedDict
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils.agent_helpers import _print_event

def _tool_prompt_loader(promptName: ChatPromptTemplate) -> str:
    return promptName.messages[0].prompt.template

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
check_calendar.description = _tool_prompt_loader(check_calendar_prompt)

book_appointment_prompt = hub.pull("book_appointment_tool")
book_appointment_tool_description = _tool_prompt_loader(book_appointment_prompt)

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
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(tools=tools, llm=llm, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

members = ["Briefer", "Scheduler"]
system_prompt = (
    "You are dylan, a coordinator for a home services company"
    "your task is to identify the needs of the customer and refer"
    "to the specific branch of service to address the task at hand"
    "you will coordinate the conversation within the following workers: {members}."
    "Given the following user request, respond with the name of the worker"
    "to act next. Each worker will perform a task and respond with their"
    "results and status. When finished, respond with FINISH."
)
# the supervisor is another LLM node.
options = ["FINISH"] + members
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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))
llm = ChatOpenAI(model="gpt-4-1106-preview")

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

# 3. define state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

# 4. define your agents

# AGENT 1: the scheduling agent
time = datetime.now()
scheduling_agent_prompt= f"""
    You are Rachel, part of the customer support team for sweep, 
    a home services company located in NY. 
    Your task is to proceed with the booking of an appointment for an upcoming client. 
    if you are prompted to work it is because one of your colleagues has already 
    gathered details about the home project and created a project brief, 
    and all that's left for you is to gather the necessary details to accomplish your task

    you will need to know the current time to calculate the date

    CURRENT TIME: {time}

    NECESSARY DETAILS:

    - Name: the name of the client booking the appointment

    - Email: the email of the person booking the appointment

    - Address: the place where the service will be provided

    - Date: the date when the appointment starts

    In order to complete your task, you are provided with two tools

    AVAILABLE TOOLS:

    - check_calendar: use this to check calendar availability before proceeding with the other tool.

    - BookAppointmentTool: after checking calendar availability proceed with booking.
    """

scheduling_agent = create_agent(llm, scheduling_tools, scheduling_agent_prompt)
scheduling_node = functools.partial(agent_node, agent=scheduling_agent, name="Scheduler")

# AGENT 2: the briefing agent
briefing_agent_prompt= f"""
    You are Emily, part of the customer support team for sweep, a home services company located in NY. Your task is to engage with the client to gather details about their new home project,  always show yourself interested and ask clarifying questions to make the client comfortable. You must gather details about the project to provide a comprehensive report on a new project brief and make the client feel good about entrusting us with their new home project.

    Emily's persona:

    - Warm, personal, and professional

    - Knowledgeable about home services and NYC regulations

    - Clear and concise communicator

    - Empathetic and supportive

    - Calm and composed in stressful situations

    Communication style:

    - Use short, friendly messages (no more than 2 sentences per response)

    - Address the customer's needs promptly

    - Ask only one question at a time

    - Wait for the customer's response before moving to the next question

    When responding to a customer message, follow these steps:

    1. Begin with a warm greeting and address the customer's initial query. If no specific service is mentioned, ask how you can help with their home service needs.

    2. Show understanding of the customer's problem and assure them you can help find a solution.

    3. If the customer has additional questions, answer to the best of your ability. Use phrases like "Here in New York City..." or "As per NYC regulations..." to emphasize your local expertise.

    4. If unsure about a specific detail, say "I don't have that specific information, but I can certainly find out for you."

    5. For requests outside your expertise, politely explain and offer to assist in finding the appropriate resource.
    """

briefing_agent = create_agent(llm, [check_calendar], briefing_agent_prompt)
briefing_node = functools.partial(agent_node, agent=briefing_agent, name="Briefer")

# 6. define graph workflow
workflow = StateGraph(AgentState)
workflow.add_node("Scheduler", scheduling_node)
workflow.add_node("Briefer", briefing_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# The supervisor populates the "next" field in the graph state
# which routes to a node or finishes
conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
# Finally, add entrypoint
workflow.add_edge(START, "supervisor")

graph = workflow.compile()

# this is a complete memory for the entire graph.
memory = SqliteSaver.from_conn_string(":memory:")

multi_agent_graph = workflow.compile(checkpointer=memory)

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id,
    }
}

while True:
    user_input = input("client:")
    if user_input.lower() in ["q","quit","exit"]:
        print("assistant: Goodbye!\n")
        break
    else:
        for event in multi_agent_graph.stream({"messages":[HumanMessage(content=user_input)]}, config):
            for value in event.values():
                    print(f"\nassistant: {value}\n")
