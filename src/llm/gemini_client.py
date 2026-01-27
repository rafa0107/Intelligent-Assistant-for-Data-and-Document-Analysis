import google.generativeai as genai
import os

def generate_answer(prompt : str) -> str:

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found")

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    # DEBUG opcional
    # print(response)

    if hasattr(response, "text") and response.text:
        return response.text.strip()

    # fallback para estrutura interna
    try:
        return response.candidates[0].content.parts[0].text.strip()
    except Exception:
        raise RuntimeError("Gemini returned no usable text")