import logging
import uuid

from twilio.rest import Client
from decouple import config
from src.db.models.conversations import Conversation
from src.agent import agent_graph
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session


account_sid = config("TWILIO_ACCOUNT_SID")
auth_token = config("TWILIO_AUTH_TOKEN")
client = Client(account_sid, auth_token)
twilio_number = config("TWILIO_NUMBER")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_message(to_number:str, body_text:str):
    try:
        message = client.messages.create(
            from_=f"whatsapp:{twilio_number}",
            body=body_text,
            to=f"whatsapp:{to_number}"
            )
        logger.info(f"Message sent to {to_number}: {message.body}")
    except Exception as e:
        logger.error(f"Error sending message to {to_number}: {e}")

def get_or_create_thread_id(db: Session, phone_number: str) -> uuid.UUID:
    conversation = db.query(Conversation).filter(Conversation.sender == phone_number).first()
    if conversation:
        return conversation.thread_id
    else:
        # Create a new conversation with a new UUID thread_id
        new_thread_id = uuid.uuid4()
        return new_thread_id


def get_agent_message(query:str, phone_number:str, thread_id:uuid.UUID) -> str:
    configurable = {
        "configurable":{
            "user_id": phone_number,
            "thread_id": thread_id,
        }
    }
    state = agent_graph.invoke({"messages": query}, configurable)
    try:
        agent_message = state["messages"][-1].content
        return agent_message
    except:
        agent_message = state["messages"][-1].tool_calls[0].content
        return agent_message

def save_conversation(db: Session, query:str, phone_number:str, thread_id:uuid.UUID, response:str) -> None:
    new_conversation = Conversation(
        sender=phone_number, 
        message=query, 
        query_response=response,
        thread_id=thread_id)
    try:
        db.add(new_conversation)
        db.commit()
        logger.info(f"Conversation #{new_conversation.id} stored in database")
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error: {e}")


def get_response(db: Session, query: str, phone_number: str) -> str:
    thread_id = get_or_create_thread_id(db, phone_number)
    response = get_agent_message(
        query=query,
        phone_number=phone_number,
        thread_id = thread_id)
    save_conversation(
        db = db,
        query = query,
        phone_number = phone_number,
        thread_id = thread_id,
        response = response)
    return response


