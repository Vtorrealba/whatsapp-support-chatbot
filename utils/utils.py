import logging
from twilio.rest import Client
from decouple import config


account_sid = config("TWILIO_ACCOUNT_SID")
auth_token = config("TWILIO_AUTH_TOKEN")
twilio_number = config("TWILIO_NUMBER")

client = Client(account_sid, auth_token)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_whatsapp(to_number:str, body_text:str):
    try:
        message = client.messages.create(
            from_=f"whatsapp:{twilio_number}",
            body=body_text,
            to=f"whatsapp:{to_number}"
            )
        logger.info(f"Message sent to {to_number}: {message.body}")
    except Exception as e:
        logger.error(f"Error sending message to {to_number}: {e}")

def send_sms(to_number:str, body_text:str):
    try:
        message = client.messages.create(
            from_=f"{twilio_number}",
            body=body_text,
            to=f"{to_number}"
            )
        logger.info(f"Message sent to {to_number}: {message.body}")
    except Exception as e:
        logger.error(f"Error sending message to {to_number}: {e}")

