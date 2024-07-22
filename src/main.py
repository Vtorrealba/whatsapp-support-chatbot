from fastapi import FastAPI, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from decouple import config
from db.base import Base
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from db.session import engine, SessionLocal 
from utils.utils import send_message, get_response, logger

Base.metadata.create_all(engine)
class Query(BaseModel):
    message:str


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
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

@app.post("/message")
async def reply(request: Request, Body: str = Form(), db: Session = Depends(get_db)):
    # Extract the phone number from the incoming webhook request
    form_data = await request.form()
    whatsapp_number = form_data['From'].split("whatsapp:")[-1]
    print(f"Sending the LangChain response to this number: {whatsapp_number}")

    # Get the generated text from the LangChain agent
    langchain_response = get_response(db, Body, whatsapp_number)
    send_message(whatsapp_number, langchain_response)
    return ""

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
