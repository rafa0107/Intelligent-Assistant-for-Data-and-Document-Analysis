import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
from sentence_transformers import SentenceTransformer

# ========= PATHS =========
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT_DIR, "src", "application"))
sys.path.append(os.path.join(ROOT_DIR, "src", "llm"))

from rag_service import RAGService  # type: ignore
from gemini_client import configure_gemini  # type: ignore


# ========= CACHE =========

@st.cache_resource
def load_model():
    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )


@st.cache_data
def load_data():
    embeddings = np.load(
        os.path.join(ROOT_DIR, "data", "processed", "embeddings.npy")
    )
    df = pd.read_csv(
        os.path.join(ROOT_DIR, "data", "processed", "issue_processed.csv")
    )
    return embeddings, df


@st.cache_resource
def load_rag_service():
    model = load_model()
    embeddings, df = load_data()

    return RAGService(
        model=model,
        embeddings=embeddings,
        dataframe=df,
        top_k=30,
        similarity_threshold=0.30,
        max_context_chars=800,
        max_documents=6
    )


# ========= UI =========

st.set_page_config(page_title="RAG Assistant", layout="centered")
st.title("📄 RAG Assistant")
st.caption("Semantic search + LLM over your documents")


# ========= API KEY INPUT =========

with st.sidebar:
    st.header("🔐 Configuração")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Sua chave não é salva. Usada apenas nesta sessão."
    )

    st.markdown(
        "[Obter uma API Key gratuita](https://ai.google.dev/)"
    )

# Bloqueia app se não houver chave
if not api_key:
    st.info("🔑 Insira sua API Key para iniciar o assistente.")
    st.stop()

# Configura Gemini uma única vez por sessão
if "gemini_configured" not in st.session_state:
    configure_gemini(api_key)
    st.session_state.gemini_configured = True


# ========= RAG =========

rag_service = load_rag_service()


# ========= CHAT =========

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input do usuário
user_input = st.chat_input("Digite sua pergunta...")

if user_input:
    # Pergunta
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Resposta
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = rag_service.answer_question(user_input)
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
