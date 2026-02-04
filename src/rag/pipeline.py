import numpy as np
import sys
import os


from src.rag.retriever import encode_query, cosine_similiarity_func, check_semantic_threshold
from src.rag.context_builder import top_k_index, context_builder
from src.rag.validator import (
    classify_question,
    validate_context,
    static_fallback,
    insufficient_context_response
)
from src.llm.gemini_client import generate_answer
from src.prompts.rag_prompts import build_prompt


def run_rag(
    question: str,
    model,
    embeddings,
    dataframe,
    top_k: int = 30,
    similarity_threshold: float = 0.30,
    max_context_chars: int = 800,
    max_documents: int = 6
) -> str:

    #  respostas estáticas
    static = static_fallback(question)
    if static:
        return static

    #  classificar pergunta
    question_type = classify_question(question)

    #  embedding da query
    query_embedding = encode_query(model, [question]) # type: ignore

    #  similaridade
    scores = cosine_similiarity_func(query_embedding, embeddings).flatten()

    #  top-k
    scores_sorted = np.argsort(scores)
    top_indices = top_k_index(top_k, scores_sorted) # type: ignore

    #  limiar semântico
    metadata = check_semantic_threshold(
        scores=scores,
        top_indices=top_indices,
        limiar=similarity_threshold,
        dataframe=dataframe
    )

    if not metadata:
        return insufficient_context_response(question_type)

    #  construir contexto
    context = context_builder(
        max_context_chars,
        metadata,
        max_documents
    )

    #  validar contexto
    if not validate_context(context):
        return insufficient_context_response(question_type)

    #  prompt
    prompt = build_prompt(
        context=context,
        question=question,
        question_type=question_type
    )

    #  LLM
    return generate_answer(prompt)
