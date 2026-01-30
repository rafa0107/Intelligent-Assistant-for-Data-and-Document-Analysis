import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import pandas as pd
import shutil
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA_DIR = os.path.join(BASE_DIR)
FILE_NAME = "github_issues.csv"
LOCAL_FILE_PATH = os.path.join(LOCAL_DATA_DIR, FILE_NAME) #coloca o path absoluto com o caminho do arquivo data/raw/github-issues.csv
nltk.download('stopwords') #carrego stopwrds para filtragem NLP nas funcoes abaixo


##CONSERTAR ESSA PARTE, NAO ESTÁ ALOCANDO CORRETAMENTE A BASE DE DADOS NA PASTA

def carregar_dados():
    if not os.path.exists(LOCAL_DATA_DIR): #criando pasta do arquivo dentro do projeto
        os.makedirs(LOCAL_DATA_DIR)
    
    if not os.path.exists(LOCAL_FILE_PATH):
        print("Baixando arquivo do Kaggle e movendo o dataset para pasta do projeto ")
        cache_path = kagglehub.dataset_download("davidshinn/github-issues")

        # Encontra o arquivo no cache e copia para pasta local
        origem = os.path.join(cache_path, FILE_NAME)
        shutil.copy(origem, LOCAL_FILE_PATH)
        print(f"Arquivo salvo em: {LOCAL_FILE_PATH}")

    return pd.read_csv(LOCAL_FILE_PATH, nrows=800)

def make_prettier(styler):
    styler.set_caption("Logs do GitHub")
    colunas_numericas = styler.data.select_dtypes(include=['number']).columns
    
    if not colunas_numericas.empty:
        styler.background_gradient(axis=None, subset=colunas_numericas, cmap="YlGnBu")
    
    return styler

def verifica_vazios(df, coluna):
    nulos = df[coluna].isna().sum() #soma a quantidade de nan na coluna especifica
    vazios = (df[coluna].str.strip() == "").sum() # soma a quantidade de vazios(ou apenas espaços) na coluna específica

    print(f"Resultados para '{coluna}':")
    print(f"- Valores Nulos (NaN): {nulos}")
    print(f"- Textos Vazios/Espaços: {vazios}")

    return df[df[coluna].isna() | (df[coluna].str.strip() == "")]


def verifica_frequentes(df, coluna, top_n, idioma='english'):
    stops = set(stopwords.words(idioma)) #retira palavras nao importantes como artigos e conectivos repetitivos nos textos
    stops.update(['https','http', 'com', 'issue', 'github', 'www']) #adicionando termos técnicos que eu nao quero que apareça

    texto_completo = " ".join(df[coluna].dropna().astype(str).str.lower())
    palavras = re.findall(r'\w+', texto_completo)
    palavras_limpas = [p for p in palavras if p not in stops and not p.isdigit()] #limpa as palavras com artigos e mais usadas para complementos
    contagem = Counter(palavras_limpas)
    return contagem.most_common(top_n)
