
def build_prompts(context:str, question:str)->str:
        return f"""
    You are a technical assistant.

    Answer the question using ONLY the context below.
    If the context is insufficient, say so clearly.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """.strip()