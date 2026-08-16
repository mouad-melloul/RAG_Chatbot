from langchain_ollama import ChatOllama

from app.config import (
    OLLAMA_BASE_URL,
    LLM_MODEL
)


class LLMManager:
    """
    Manages the Ollama language model.
    """

    def __init__(self):
        self.model = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL
        )

    def get_model(self):
        """
        Return the configured language model.
        """

        return self.model