from datetime import datetime
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage 
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict
from typing import Annotated
from src.tools.scheduling import check_calendar, book_appointment


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


tools = [check_calendar, book_appointment]


# define state
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    

# assistant node
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
            
    
    
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)


# bind tools node to llm
agent_prompt = hub.pull("customer_support_chatbot").partial(time=datetime.now().strftime("%m/%d/%Y"))
agent_runnable = agent_prompt | llm.bind_tools(tools)

# define graph workflow
builder = StateGraph(State)

# define nodes: these do the work
builder.add_node("assistant", Assistant(agent_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))


# define edges: these determine how the control flow moves
builder.set_entry_point("assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

memory = SqliteSaver.from_conn_string(":memory:")
agent_graph = builder.compile(checkpointer=memory)

