from sentence_transformers import SentenceTransformer
from pipeline import run_rag

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2",
    device="cpu"
)

resposta = run_rag(
    question="Quais são os erros mais registrados no documento?",
    model=model,
    embeddings=embeddings,
    dataframe=df,
    top_k=30,
    similarity_threshold=0.30,
    max_context_chars=800,
    max_documents=6
)

print(resposta)
