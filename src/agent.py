import os 
import dotenv
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import ToolNode

dotenv.load_dotenv()

# 1. setup observability and env vars
os.environ['LANGCHAIN_PROJECT'] = "Sweep chatbot"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2. define tools
requests_post = load_tools(["requests_post"], allow_dangerous_tools=True)
tools = [requests_post]
tool_node = ToolNode(tools)


model = ChatOpenAI(temperature=0.9, model_name="gpt-4-turbo").bind_tools(tool_node)

print(requests_post)
class Agent:
    def __init__(self):
        self.agent = None
        self.memory = []
        self.context = []
        self.response = ""
        self.user_input = ""
        self.assistant_input = ""

    def get_response(self, user_input):
        self.user_input = user_input
        self.assistant_input = self.agent.run(self.user_input)
        return self.assistant_input
    