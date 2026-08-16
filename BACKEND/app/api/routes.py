import os
import shutil

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException
)

from pydantic import BaseModel

from app.config import (
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS
)

from app.rag.pipeline import RAGPipeline


router = APIRouter(
    prefix="/api"
)


# ---------------------------------------------------------
# Global RAG pipeline
# ---------------------------------------------------------

rag_pipeline = RAGPipeline()


# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def allowed_file(filename: str):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@router.get("/health")
def health():

    return {
        "status": "ok"
    }


# ---------------------------------------------------------
# Upload PDF
# ---------------------------------------------------------

@router.post("/upload")
async def upload_files(
    pdf_file: list[UploadFile] = File(...)
):

    os.makedirs(
        UPLOAD_FOLDER,
        exist_ok=True
    )

    results = []

    for file in pdf_file:

        if not file.filename:
            continue

        if not allowed_file(file.filename):

            results.append({
                "file": file.filename,
                "status": "error",
                "message": "Only PDF files are allowed."
            })

            continue

        # Safe filename
        filename = os.path.basename(
            file.filename
        )

        file_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        try:

            # Save file
            with open(
                file_path,
                "wb"
            ) as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer
                )

            print(
                f"File saved: {file_path}"
            )

            # Process PDF
            chunks_count = (
                rag_pipeline.add_document(
                    file_path
                )
            )

            results.append({
                "file": filename,
                "status": "ok",
                "chunks": chunks_count
            })

        except Exception as e:

            print(
                f"Error processing {filename}: {e}"
            )

            results.append({
                "file": filename,
                "status": "error",
                "message": str(e)
            })

    if not results:

        raise HTTPException(
            status_code=400,
            detail="No valid PDF file was uploaded."
        )

    return {
        "message": "Upload completed.",
        "files": results
    }


# ---------------------------------------------------------
# Ask
# ---------------------------------------------------------

@router.post("/ask")
def ask_question(
    request: QuestionRequest
):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        answer = rag_pipeline.ask(
            question
        )

        return {
            "answer": answer
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        print(
            f"Error answering question: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the answer."
        )


# ---------------------------------------------------------
# Clear
# ---------------------------------------------------------

@router.post("/clear")
def clear_conversation():

    rag_pipeline.clear()

    return {
        "message": "Conversation cleared."
    }