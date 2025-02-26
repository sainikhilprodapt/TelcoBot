import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = "gemini-1.5-pro"
    EMBEDDING_MODEL = "models/embedding-001"
    DB_CONNECTION = os.getenv("DB_CONNECTION")