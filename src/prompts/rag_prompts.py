def build_prompts(context: str, question: str) -> str:
    return f"""
You are a technical assistant.

Answer the question using the context below.
You may infer patterns or repeated issues IF they are clearly present in the context.
Do NOT invent errors that are not mentioned.
If the context truly lacks information, say so.

Context:
{context}

Question:
{question}

Answer:
""".strip()