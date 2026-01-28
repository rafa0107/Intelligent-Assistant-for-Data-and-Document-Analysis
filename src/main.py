import sys
import os
import nltk
from nltk.corpus import stopwords
from dotenv import load_dotenv

from dataset import verifica_vazios, verifica_frequentes

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd #ler diretamente esta célula sem precisar converter novamente os arquivos no embedding

nltk.download('stopwords') #carrego stopwrds para filtragem NLP

# Adiciona o caminho da pasta onde o arquivo 'dataset.py' está localizado
# Isso pula a necessidade de mencionar a pasta com hífen no import
caminho_raw = os.path.abspath(os.path.join('..', 'data', 'raw'))
if caminho_raw not in sys.path:
    sys.path.append(caminho_raw)

from dataset import carregar_dados

df = carregar_dados()
print(df.head())


sys.path.append(os.path.abspath(os.path.join('..', 'data', 'raw')))
sys.path.append(os.path.abspath(os.path.join('..', 'data', 'processed')))
sys.path.append(os.path.abspath(os.path.join('..', 'src', 'prompts')))
sys.path.append(os.path.abspath(os.path.join('..', 'src', 'llm')))

from data_processing import clean_text
from rag_prompts import build_prompts
from gemini_client import generate_answer
from semantic_search import encode_query, cosine_similiarity_func


df["clean_title"] = df["issue_title"].apply(clean_text)
df["clean_body"] = df["body"].apply(clean_text)


media_palavras_url = df["issue_url"].str.split().str.len().mean()
media_palavras_title = df["clean_title"].str.split().str.len().mean()
media_palavras_body = df["clean_body"].str.split().str.len().mean()

print(f"Média de palavras em url: {media_palavras_url}, em title: {media_palavras_title}, em body: {media_palavras_body}")

url_vazio = verifica_vazios(df,"issue_url")
titles_vazio = verifica_vazios(df,"clean_title")
body_vazio = verifica_vazios(df,"clean_body")

palavras_frequentes_titles = verifica_frequentes(df,"clean_title", 20)
print(f"Palavras mais frequentes em titles: {palavras_frequentes_titles}")
palavras_frequentes_body = verifica_frequentes(df,"clean_body", 20)
print(f"Palavras mais frequentes em body: {palavras_frequentes_body}")



#criação de nova coluna para texto final que será direcionado ao embedding
df["final_text"] = (
    "Title: " + df["clean_title"] +
    ". Body: " + df["clean_body"]
)

#carregando o modelo que será usado, modelo rápido e leve para projeto OBS: Uso da CPU pois GPU esta ultrapassada para o modelo
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu") #usando o paraphrasal para poder perguntar em portugues
df[["clean_title", "clean_body", "final_text"]].head()



#listando o texto final em variavel para ser codificada
texts = df["final_text"].tolist()

#realização do embedding pelo modelo escolhido
embeddings = model.encode( 
    texts,
    show_progress_bar= True
)

#criação do array em np 
embeddings = np.array(embeddings)
print(embeddings.shape)

#persistindo os dados em formato npy e csv para nao necessitar de conversao novamente
np.save("embeddings.npy",embeddings)
df.to_csv("issue_processed.csv", index=False)




embeddings = np.load("embeddings.npy")
df = pd.read_csv("issue_processed.csv")



pergunta = "Quais sao os erros mais registrados no documento?"
query = encode_query(model, [pergunta])
#print(query.shape)

scores = cosine_similiarity_func(query,embeddings)
scores = scores.flatten()
scores_ordenados = np.argsort(scores)
print(scores.shape)
print(scores_ordenados)

#fazendo busca top_k, depois refatorar montando em funcoes definidas
# Atualizar o Readme e usar o topk como parametro para busca ao refatorar “Utilizamos top-k dinâmico para balancear cobertura semântica e precisão.”
top_k = 3
top_indices = scores_ordenados[-top_k:][::-1] #quero buscar os ultimos 20 valores e ordena-los em sequência maior para menor, pois np.arg retorna ordem crescente
print(top_indices)


#definir o limiar, fazer um loop retornando os melhores índices
limiar = 0.30
lista_final = []

for i in range(len(top_indices)):
    if scores[top_indices[i]] > limiar:
        lista_final.append({"indice" : top_indices[i] ,"score" :scores[top_indices[i]], "text" : df.iloc[top_indices[i]]["final_text"]})

print(lista_final)


MAX_CHARS = 400  # por documento

lista_final = sorted(
    lista_final,
    key=lambda x: x["score"],
    reverse=True
)

contexto = ""
limite = 3

if len(lista_final) <= limite:
    for i in range(len(lista_final)):
        text_limited = lista_final[i]["text"][:MAX_CHARS]

        contexto += (
            f"[Contexto {i+1} | "
            f"Similaridade: {lista_final[i]['score']:.03f}]\n"
            f"{text_limited}\n\n"
        )

else:
    for i in range(limite):
        text_limited = lista_final[i]["text"][:MAX_CHARS]

        contexto += (
            f"[Contexto {i+1} | "
            f"Similaridade: {lista_final[i]['score']:.03f}]\n"
            f"{text_limited}\n\n"
        )

print(contexto)

load_dotenv()
chave_api = os.getenv("GOOGLE_API_KEY")

#prompt = build_prompts(contexto, pergunta)
#resposta = generate_answer(prompt)

#print(resposta)

# Tente um prompt simples para validar a chave
#teste = generate_answer("Olá, você está funcionando?")
#print(teste)