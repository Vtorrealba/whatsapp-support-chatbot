from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Annotated
from pydantic import BaseModel
from whatsapp_support_chatbot.agent import Agent

class Query(BaseModel):
    message:str
    
agent = Agent()

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"undefined": "this is not a valid endpoint, try /api/v1/chatbot with a POST request"}
  
@app.post("/api/v1/chatbot")
async def chatbot(query:Query):
    response = agent.get_response(query.message)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
