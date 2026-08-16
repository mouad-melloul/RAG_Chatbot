import faiss

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL,
    TOP_K,
    FETCH_K,
    LAMBDA_MULT
)

from app.rag.document_processor import DocumentProcessor


class RAGPipeline:

    def __init__(self):

        self.vector_store = None

        # -------------------------------------------------
        # Embeddings
        # -------------------------------------------------

        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )

        # -------------------------------------------------
        # LLM
        # -------------------------------------------------

        self.model = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL
        )

        # -------------------------------------------------
        # Document processor
        # -------------------------------------------------

        self.document_processor = DocumentProcessor()

        # -------------------------------------------------
        # Prompt
        # -------------------------------------------------

        prompt_template = """
You are a helpful and accurate document assistant.

Your job is to answer questions based strictly on
the context provided below.

Guidelines:
- Answer clearly and directly using only what is written
  in the context.
- If the answer is a list, present it in a clean numbered
  or bulleted format.
- Do not add commentary, assumptions, or analysis beyond
  what was asked.
- Do not use filler phrases like "it appears" or
  "based on the context".
- If the information is not in the context, simply say:
  "This information is not mentioned in the document."
- Keep answers focused.
- Do not add unrequested sections or conclusions.

Context:
{context}

Question:
{question}

Answer:
"""

        self.prompt = ChatPromptTemplate.from_template(
            prompt_template
        )

    # =====================================================
    # ADD PDF
    # =====================================================

    def add_document(self, file_path: str):

        print(f"Processing PDF: {file_path}")

        # Extract and split
        chunks = self.document_processor.process(
            file_path
        )

        print(
            f"Generated {len(chunks)} chunks."
        )

        # -------------------------------------------------
        # Create FAISS store for this document
        # -------------------------------------------------

        sample_vector = self.embeddings.embed_query(
            "sample text"
        )

        dimension = len(sample_vector)

        index = faiss.IndexFlatL2(dimension)

        new_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

        # Add chunks
        ids = new_store.add_documents(
            documents=chunks
        )

        print(
            f"Added {len(ids)} chunks to FAISS."
        )

        # -------------------------------------------------
        # Merge with existing documents
        # -------------------------------------------------

        if self.vector_store is None:

            self.vector_store = new_store

            print(
                "Created global vector store."
            )

        else:

            self.vector_store.merge_from(
                new_store
            )

            print(
                "Merged document into global vector store."
            )

        return len(chunks)

    # =====================================================
    # CREATE RETRIEVER
    # =====================================================

    def get_retriever(self):

        if self.vector_store is None:
            raise ValueError(
                "No documents have been uploaded yet."
            )

        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": TOP_K,
                "fetch_k": FETCH_K,
                "lambda_mult": LAMBDA_MULT
            }
        )

    # =====================================================
    # ASK QUESTION
    # =====================================================

    def ask(self, question: str):

        if self.vector_store is None:
            raise ValueError(
                "Please upload a PDF file first."
            )

        retriever = self.get_retriever()

        def format_docs(docs):

            return "\n\n".join(
                doc.page_content
                for doc in docs
            )

        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        response = rag_chain.invoke(
            question
        )

        return response

    # =====================================================
    # CLEAR
    # =====================================================

    def clear(self):

        self.vector_store = None