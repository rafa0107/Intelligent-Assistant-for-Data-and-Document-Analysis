# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import pandas as pd
import shutil

# Set the path to the file you'd like to load
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA_DIR = os.path.join(BASE_DIR)
FILE_NAME = "github_issues.csv"
LOCAL_FILE_PATH = os.path.join(LOCAL_DATA_DIR, FILE_NAME) #coloca o path absoluto com o caminho do arquivo data/raw/github-issues.csv

def carregar_dados():
    if not os.path.exists(LOCAL_DATA_DIR): #criando pasta do arquivo dentro do projeto
        os.makedirs(LOCAL_DATA_DIR)
    
    if not os.path.exists(LOCAL_FILE_PATH):
        print("Baixando arquivo do Kaggle e movendo o dataset para pasta do projeto ")
        cache_path = kagglehub.dataset_download("davidshinn/github-issues")

        # Encontra o arquivo no cache e copia para sua pasta local
        origem = os.path.join(cache_path, FILE_NAME)
        shutil.copy(origem, LOCAL_FILE_PATH)
        print(f"Arquivo salvo em: {LOCAL_FILE_PATH}")

    return pd.read_csv(LOCAL_FILE_PATH, nrows=100)

df = carregar_dados()
print(df.head())