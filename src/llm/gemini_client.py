import google.genai as genai
import os
import time

def generate_answer(prompt: str, max_retries: int = 2) -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY não encontrada.")

    client = genai.Client(api_key=api_key) # type: ignore

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text.strip() # type: ignore

        except Exception as e:
            return f"Erro ao gerar resposta: {e}"

    return "Falha ao gerar resposta após múltiplas tentativas."
