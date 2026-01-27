
def build_prompts(context:str, question:str)->str:
    prompt = f"""
    You are an AI assistant specialized in analyzing technical documentation.

    Use ONLY the information provided in the context below to answer the user's question.
    If the answer is not clearly supported by the context, say that there is not enough information to answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    return prompt.strip()