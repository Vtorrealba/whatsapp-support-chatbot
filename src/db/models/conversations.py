from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import UUID
import uuid
from src.db.base import Base

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    sender = Column(String)
    message = Column(String)
    response = Column(String)
    thread_id = Column(UUID(as_uuid=True), index=True)
    


