import os
import sys
from typing import Any
from src.rag.pipeline import run_rag

class RAGService:
    def __init__(
        self,
        model: Any,
        embeddings,
        dataframe,
        top_k: int = 30,
        similarity_threshold: float = 0.30,
        max_context_chars: int = 800,
        max_documents: int = 6,
    ):
        self.model = model
        self.embeddings = embeddings
        self.dataframe = dataframe
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.max_context_chars = max_context_chars
        self.max_documents = max_documents

    def answer_question(self, question: str) -> str:
        return run_rag(
            question=question,
            model=self.model,
            embeddings=self.embeddings,
            dataframe=self.dataframe,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold,
            max_context_chars=self.max_context_chars,
            max_documents=self.max_documents,
        )
