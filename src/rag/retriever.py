import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def encode_query(model, query: str):
    return model.encode(query)

def cosine_similiarity_func(query, base): 
    return cosine_similarity(query,base)

def check_semantic_threshold(scores, top_indices, limiar, dataframe)->list:
    lista_final = []

    for i in range(len(top_indices)):
        if scores[top_indices[i]] > limiar:
            lista_final.append({"indice" : top_indices[i] ,"score" :scores[top_indices[i]], "text" : dataframe.iloc[top_indices[i]]["final_text"]})

    return lista_final