import logging
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import URL
from decouple import config

# Load environment variables
DB_USER = config("DB_USER")
DB_PASSWORD = config("DB_PASSWORD")
DB_HOST = config("DB_HOST")
DB_PORT = config("DB_PORT")
DB_NAME = config("DB_NAME")

# Setup the connection URL
url = URL.create(
    drivername="postgresql",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)

# Create the engine
engine = create_engine(url, echo=True)

# Create session local
SessionLocal = sessionmaker(bind=engine)

# Enable logging for SQLAlchemy
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Log raw SQL statements
logging.getLogger('sqlalchemy.dialects').setLevel(logging.DEBUG)


    