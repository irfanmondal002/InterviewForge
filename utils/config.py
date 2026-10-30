import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_DEFAULT_MODEL = os.getenv(
    "GEMINI_DEFAULT_MODEL",
    "gemini-3.6-flash"
)


# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

GROQ_DEFAULT_MODEL = os.getenv(
    "GROQ_DEFAULT_MODEL",
    "openai/gpt-oss-120b"
)


# Ollama
OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434"
)

OLLAMA_DEFAULT_MODEL = os.getenv(
    "OLLAMA_DEFAULT_MODEL",
    ""
)


# Embeddings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Storage
DATA_DIR = PROJECT_ROOT / "data"

DB_PATH = DATA_DIR / "interviewforge.db"

CHROMA_PATH = PROJECT_ROOT / ".chroma"


DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHROMA_PATH.mkdir(
    parents=True,
    exist_ok=True
)