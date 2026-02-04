import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src', 'rag')))
sys.path.append(os.path.abspath(os.path.join('..', 'src', 'llm')))
sys.path.append(os.path.abspath(os.path.join('..', 'src', 'prompts')))

from retriever import encode_query, cosine_similiarity_func, check_semantic_threshold
from context_builder import top_k_index, context_builder
from validator import (
    classify_question,
    validate_context,
    static_fallback,
    insufficient_context_response
)
from gemini_client import generate_answer
from rag_prompts import build_prompt


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

    # 1️⃣ respostas estáticas
    static = static_fallback(question)
    if static:
        return static

    # 2️⃣ classificar pergunta
    question_type = classify_question(question)

    # 3️⃣ embedding da query
    query_embedding = encode_query(model, [question]) # type: ignore

    # 4️⃣ similaridade
    scores = cosine_similiarity_func(query_embedding, embeddings).flatten()

    # 5️⃣ top-k
    scores_sorted = np.argsort(scores)
    top_indices = top_k_index(top_k, scores_sorted) # type: ignore

    # 6️⃣ limiar semântico
    metadata = check_semantic_threshold(
        scores=scores,
        top_indices=top_indices,
        limiar=similarity_threshold,
        dataframe=dataframe
    )

    if not metadata:
        return insufficient_context_response(question_type)

    # 7️⃣ construir contexto
    context = context_builder(
        max_context_chars,
        metadata,
        max_documents
    )

    # 8️⃣ validar contexto
    if not validate_context(context):
        return insufficient_context_response(question_type)

    # 9️⃣ prompt
    prompt = build_prompt(
        context=context,
        question=question,
        question_type=question_type
    )

    # 🔟 LLM
    return generate_answer(prompt)
