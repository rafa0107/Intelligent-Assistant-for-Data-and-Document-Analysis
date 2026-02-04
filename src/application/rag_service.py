import google.genai as genai
import os
import sys

sys.path.append(os.path.abspath(os.path.join('..', 'src', 'llm')))
sys.path.append(os.path.abspath(os.path.join('..', 'src', 'prompts')))
sys.path.append(os.path.abspath(os.path.join('..', 'src', 'rag')))
from validator import validate_context, static_fallback, insufficient_context_response 
from rag_prompts import classify_question, build_prompt
from gemini_client import generate_answer

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