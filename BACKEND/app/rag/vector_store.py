"""
- Ollama embeddings
- FAISS
- adding documents
- merging multiple PDFs
- creating the retriever
"""

import faiss

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from app.config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    RETRIEVER_K,
    RETRIEVER_FETCH_K,
    RETRIEVER_LAMBDA
)


class VectorStoreManager:
    """
    Manages embeddings, FAISS vector store,
    and document retrieval.
    """

    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )

        self.vector_store = None

    def create_store(self, documents):
        """
        Create a new FAISS vector store from documents.
        """

        # Get embedding dimension
        sample_embedding = self.embeddings.embed_query(
            "this is some text data"
        )

        print(
            f"Embedding dimension: {len(sample_embedding)}"
        )

        # Create FAISS index
        index = faiss.IndexFlatL2(
            len(sample_embedding)
        )

        new_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

        # Add documents
        ids = new_store.add_documents(
            documents=documents
        )

        print(
            f"Documents added to vector store. IDs: {ids}"
        )

        return new_store

    def add_documents(self, documents):
        """
        Add documents to the global vector store.

        If no vector store exists, create one.
        Otherwise, merge the new store with the existing one.
        """

        new_store = self.create_store(documents)

        if self.vector_store is None:

            self.vector_store = new_store

            print("Global vector store created.")

        else:

            self.vector_store.merge_from(new_store)

            print(
                "New documents merged into existing vector store."
            )

    def get_retriever(self):
        """
        Return a retriever based on the current vector store.
        """

        if self.vector_store is None:
            raise ValueError(
                "Vector store is empty. Upload a PDF first."
            )

        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": RETRIEVER_K,
                "fetch_k": RETRIEVER_FETCH_K,
                "lambda_mult": RETRIEVER_LAMBDA
            }
        )

        return retriever

    def is_ready(self):
        """
        Check whether documents have been indexed.
        """

        return self.vector_store is not None