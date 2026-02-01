import streamlit as st

st.title("🧠 Intelligent Document Assistant")

question = st.text_input("Digite sua pergunta:")

if st.button("Buscar"):
    resposta = "resposta"
    st.write(resposta)
