import uuid
import re
from fastapi import FastAPI, Form, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from db.session import engine, SessionLocal
from db.base import Base
from db.models.conversations import Conversation
from utils.utils import send_message, logger
from src.agent import agent_graph
from src.multi_agent import multi_agent_graph 

class Configurable(BaseModel):
    phone_number: str = Field(...)
    thread_id: uuid.UUID = Field(...)

    @field_validator('phone_number')
    @classmethod
    def validate_phone_number(cls, value):
        # Improved regular expression to validate E.164 format
        e164_pattern = re.compile(r'^\+[1-9]\d{1,14}$')
        if not e164_pattern.match(value):
            raise ValueError('Invalid phone number format. It must be in E.164 format: +[country code][number]')
        
        # Additional check for minimum length
        if len(value) < 11:  # +, country code (at least 1 digit), and at least 9 digits
            raise ValueError('Phone number is too short. It must have at least 9 digits after the country code.')
        
        return value

    @field_validator('thread_id')
    @classmethod
    def validate_thread_id(cls, value):
        if not isinstance(value, uuid.UUID):
            raise ValueError('Invalid UUID format for thread_id')
        return value

class Query(BaseModel):
    message: str

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://sweep.ngrok.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_or_create_thread_id(db:Session, phone_number: str) -> uuid.UUID:
    try:
        conversation = db.query(Conversation).filter(Conversation.sender == phone_number).first()
        if conversation:
            return conversation.thread_id
        else:
            # Create a new conversation with a new UUID thread_id
            new_thread_id = uuid.uuid4()
            return new_thread_id
    except SQLAlchemyError as e:
        logger.error(f"An error occurred while retrieving the conversation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

def build_config(phone_number: str, thread_id: uuid.UUID) -> dict:
    try:
        # Validate inputs using Configurable
        config = Configurable(phone_number=phone_number, thread_id=thread_id)
        
        # Construct the desired output format
        return {
            "configurable": {
                "user_id": config.phone_number,
                "thread_id": str(config.thread_id),  # Convert UUID back to string
            }
        }
    except ValueError as e:
        # Handle validation errors
        raise ValueError(f"Configuration error: {str(e)}")

def get_agent_message(query:str, phone_number:str, thread_id:uuid.UUID) -> str:
    config = build_config(phone_number, thread_id)
    state = agent_graph.invoke({"messages": query}, config) # query in bare string
    try:
        agent_message = state["messages"][-1].content
        return agent_message
    except:
        agent_message = state["messages"][-1].tool_calls[0].content
        return agent_message
def get_multi_agent_message(query:str , phone_number:str, thread_id:uuid.UUID) -> str:
    config = build_config(phone_number, thread_id)
    state = multi_agent_graph.invoke({"messages": [query]}, config) # query inside list
    agent_message = state["messages"][-1].content
    return agent_message

def save_conversation(db: Session, query:str, phone_number:str, thread_id:uuid.UUID, response:str) -> None:
    new_conversation = Conversation(
        sender=phone_number, 
        message=query, 
        response=response,
        thread_id=thread_id)
    try:
        db.add(new_conversation)
        db.commit()
        logger.info(f"Conversation #{new_conversation.id} stored in database")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"An error occurred while saving the conversation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

def get_response(db: Session, query: str, phone_number: str) -> str:
    thread_id = get_or_create_thread_id(db, phone_number)
    response = get_multi_agent_message( # to invoke legacy use: get_agent_message
        query=query,
        phone_number=phone_number,
        thread_id=thread_id)
    save_conversation(
        db=db,
        query=query,
        phone_number=phone_number,
        thread_id=thread_id,
        response=response)
    return response

@app.post("/message")
async def reply(request: Request, Body: str = Form(), db: Session = Depends(get_db)):
    form_data = await request.form()
    whatsapp_number = form_data['From'].split("whatsapp:")[-1]
    logger.info(f"Received message from {whatsapp_number}")
    
    try:
        langchain_response = get_response(db, Body, whatsapp_number)
        send_message(whatsapp_number, langchain_response)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing message from {whatsapp_number}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
