import os
import warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import UPLOAD_FOLDER


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------

app = FastAPI(
    title="RAG Chatbot API",
    description="Document-based RAG chatbot using FAISS and Ollama.",
    version="1.0.0"
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

app.include_router(router)


# ---------------------------------------------------------
# Root
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "RAG Chatbot API is running.",
        "docs": "/docs"
    }


# ---------------------------------------------------------
# Startup
# ---------------------------------------------------------

@app.on_event("startup")
def startup():

    os.makedirs(
        UPLOAD_FOLDER,
        exist_ok=True
    )