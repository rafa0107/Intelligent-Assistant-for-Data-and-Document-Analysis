import google.genai as genai
import os
import time
import sys
from src.rag.validator import validate_context, static_fallback, insufficient_context_response
from src.prompts.rag_prompts import classify_question, build_prompt
from dotenv import load_dotenv
from typing import Optional

_client: Optional[genai.Client] = None


def configure_gemini(api_key: str) -> None:
    global _client
    _client = genai.Client(api_key=api_key)  # type: ignore


def generate_answer(prompt: str) -> str:
    if _client is None:
        raise RuntimeError(
            "Cliente Gemini não configurado. "
            "Chame configure_gemini(api_key) antes."
        )

    try:
        response = _client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text.strip()  # type: ignore

    except Exception as e:
        return f"Erro ao gerar resposta: {e}"

def answer_question(question: str, context: str) -> str:
    # 1. respostas estáticas
    static = static_fallback(question)
    if static:
        return static

    # 2. classificar pergunta
    question_type = classify_question(question)

    # 3. validar contexto
    if not validate_context(context):
        return insufficient_context_response(question_type)

    # 4. construir prompt
    prompt = build_prompt(
        context=context,
        question=question,
        question_type=question_type
    )

    return generate_answer(prompt)
