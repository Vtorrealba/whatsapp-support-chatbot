import uuid
import dotenv
import os

from datetime import datetime
from langchain_anthropic  import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain import hub
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_core.prompts import ChatPromptTemplate 

dotenv.load_dotenv()
os.environ['LANGCHAIN_PROJECT'] = "Sweep chatbot"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")



client = Client()

llm = ChatOpenAI(model="gpt-4o", temperature=1)
assistant_prompt = hub.pull("testing_customer_support_chatbot").partial(time=datetime.now(), user_id="21458856")
assistant = assistant_prompt | llm
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id,
    }
}
def predict_llm_answer(example: dict) -> dict:
  """
  predict the llm response for answer evaluation
  """
  msg = {"input": ("user", example["input"])}
  message = assistant.invoke(msg, config)
  return {"response": message.content}

#grade prompt s

grade_prompt_answer_accuracy = hub.pull("h20/rag-answer-vs-reference")
def answer_evaluator(run, example) -> dict:
  """
  parsing evaluator for the check_calendar tool
  """
  input_question = example.inputs["input"]
  reference = example.outputs["output"]
  prediction = run.outputs["response"]
  
  grader_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
  
  answer_grader = grade_prompt_answer_accuracy | grader_llm
  
  score = answer_grader.invoke({"input":input_question,
                                "output": reference,
                                "student_answer": prediction,
                                "time": datetime.now().strftime("%m/%d/%Y")})
  
  score = score["Score"]
  
  return {"key": "answer_v_reference_score", "score": score}
    
datasets = [
  "Sweep Unit Tests - Service Details",
  "Sweep Unit Tests - Compliance",
  "Sweep Unit Tests - Cost Estimates",
  "Sweep Unit Tests - Discounts",
  "Sweep Unit Tests - Feedback + Complaints",
  "Sweep Unit Tests - Service Availability",
  "Sweep Unit Tests - general scenerios",
  "sweep_appointment_email",
  "Sweep_evals_calendar_check"
]

for dataset in datasets:
  for i in range(1,4):
    evaluate(
      predict_llm_answer,
      data=dataset,
      evaluators=[answer_evaluator],
      experiment_prefix=f"{dataset}_test {i}",
    )



  
       
        
          
            
              
               
                 
                   
                    
                      
                       
                        
                           
