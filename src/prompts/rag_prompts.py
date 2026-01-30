def classify_question(question:str)->str:
    q = question.lower()

    out_of_scope_keywords = [
        "quantos", "porcentagem", "percentual",
        "frequência exata", "quantidade",
        "como corrigir", "como resolver",
        "causa", "impacto", "solução"
    ]

    qualitative_keywords = [
        "mais comuns", "mais registrados",
        "mais frequentes", "principais"
    ]

    # Tipo C — fora do escopo
    if any(k in q for k in out_of_scope_keywords):
        return "OUT_OF_SCOPE"

    # Tipo B — qualitativa
    if any(k in q for k in qualitative_keywords):
        return "QUALITATIVE"

    # Tipo A — direta
    return "DIRECT"



def build_direct_prompt(context: str, question: str) -> str:
    return f"""
You are a technical assistant.

Answer the question using ONLY the information explicitly present in the context.
Do not infer statistics or frequency.
Do not add external knowledge.

Context:
{context}

Question:
{question}

Answer:
""".strip()

def build_qualitative_prompt(context: str, question: str) -> str:
    return f"""
You are a technical assistant analyzing software issue reports.

Using ONLY the context below, identify patterns, recurring themes, or categories.
You may describe what appears frequently or repeatedly,
but DO NOT invent numbers, counts, or percentages.

If multiple types of issues appear, group them and describe them qualitatively.

Context:
{context}

Question:
{question}

Answer:
""".strip()


def build_out_of_scope_prompt(question: str) -> str:
    return f"""
The user's question requires quantitative data or information
that is not available in the retrieved documents.

Question:
{question}

Answer:
The available documents do not provide sufficient information to answer this question accurately.
""".strip()


def build_prompt(context: str, question: str, question_type: str) -> str:
    if question_type == "DIRECT":
        return build_direct_prompt(context, question)

    if question_type == "QUALITATIVE":
        return build_qualitative_prompt(context, question)

    if question_type == "OUT_OF_SCOPE":
        return build_out_of_scope_prompt(question)

    raise ValueError("Unknown question type")