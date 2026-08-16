import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# Ollama
# ---------------------------------------------------------

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3.2:1b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)

# ---------------------------------------------------------
# RAG
# ---------------------------------------------------------

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

TOP_K = 5
FETCH_K = 20
LAMBDA_MULT = 0.5

# ---------------------------------------------------------
# Uploads
# ---------------------------------------------------------

UPLOAD_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "temp_uploads"
)

ALLOWED_EXTENSIONS = {"pdf"}