import functools
import operator
from datetime import datetime
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import ToolMessage, HumanMessage, BaseMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph, START
from typing import Annotated, Sequence, TypedDict
from src.tools.scheduling import check_calendar, book_appointment, create_brief


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


# pack the tools
scheduling_tools = [check_calendar, book_appointment, create_brief]


# define helper that facilitates the creation of the agent
def create_agent(llm:ChatOpenAI, tools: list, system_prompt:str) -> AgentExecutor:
    prompt = hub.pull(system_prompt)
    if system_prompt == "sweep_scheduling":
        prompt = hub.pull(system_prompt).partial(time=datetime.now().strftime("%Y-%m-%d"))

    agent = create_openai_tools_agent(tools=tools, llm=llm, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


members = ["NewJob", "Scheduler"]
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

prompt = hub.pull("sweep_agent_routing").partial(options=str(options))

llm = ChatOpenAI(model="gpt-4o-mini")

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
sllm = ChatOpenAI(model="gpt-4o")
scheduling_agent = create_agent(sllm, scheduling_tools, "sweep_scheduling")
scheduling_node = functools.partial(agent_node, agent=scheduling_agent, name="Scheduler")

# AGENT 2: the NewJob agent
Nllm = ChatOpenAI(model="gpt-4o")
NewJob_agent = create_agent(Nllm, [check_calendar], "sweep_briefing")
NewJob_node = functools.partial(agent_node, agent=NewJob_agent, name="NewJob")

# 6. define graph workflow
workflow = StateGraph(AgentState)
workflow.add_node("Scheduler", scheduling_node)
workflow.add_node("NewJob", NewJob_node)
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
