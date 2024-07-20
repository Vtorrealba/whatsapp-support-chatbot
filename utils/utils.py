import logging
import os

from twilio.rest import Client
from decouple import config
from src.agent import part_1_graph


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


thread_id = 41
configurable= {
    "configurable": {
        "user_id": "21458856",
        "thread_id":thread_id,
    }
}
def get_response(query:str) -> str:
    state = part_1_graph.invoke({"messages": query}, configurable)
    try:
        response = state["messages"][-1].content
    except:
        response = state["messages"][-1].tool_calls[0].content
    return response



