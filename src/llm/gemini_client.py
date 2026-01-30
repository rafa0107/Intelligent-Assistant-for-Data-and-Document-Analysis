import google.genai as genai
import os
import time
import sys

sys.path.append(os.path.abspath(os.path.join('..', 'src', 'llm')))
from context_validator import validate_context, static_fallback, insufficient_context_response
from rag_prompts import classify_question, build_prompt


def generate_answer(prompt: str, max_retries: int = 2) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY não encontrada.")

    client = genai.Client(api_key=api_key) # type: ignore

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )
            return response.text.strip() # type: ignore

        except Exception as e:
            return f"Erro ao gerar resposta: {e}"

    return "Falha ao gerar resposta após múltiplas tentativas."

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
