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

def validate_context(context: str, min_chars: int = 200) -> bool:
    """
    Verifica se o contexto é minimamente informativo.
    """
    if not context:
        return False

    if len(context.strip()) < min_chars:
        return False

    return True

def static_fallback(question: str) -> str | None:
    q = question.lower()

    if "o que é" in q and "projeto" in q:
        return (
            "Este assistente utiliza busca semântica e modelos de linguagem "
            "para analisar documentos técnicos e responder perguntas com base "
            "exclusivamente no conteúdo recuperado."
        )

    if "como funciona" in q:
        return (
            "O sistema realiza uma busca semântica nos documentos relevantes "
            "e utiliza um modelo de linguagem para sintetizar a resposta."
        )

    return None

def insufficient_context_response(question_type: str) -> str:
    if question_type == "QUALITATIVE":
        return (
            "O contexto recuperado não é suficiente para identificar padrões "
            "ou recorrência de forma confiável."
        )

    return (
        "Os documentos recuperados não contêm informações suficientes "
        "para responder a essa pergunta."
    )
