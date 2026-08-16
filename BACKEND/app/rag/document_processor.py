from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process(self, file_path: str):

        if not Path(file_path).exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        # Load PDF
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()

        # Split into chunks
        chunks = self.text_splitter.split_documents(pages)

        # Remove empty chunks
        chunks = [
            chunk
            for chunk in chunks
            if chunk.page_content.strip()
        ]

        if not chunks:
            raise ValueError(
                "No valid text could be extracted from the PDF."
            )

        return chunks