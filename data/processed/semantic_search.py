import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity






def encode_query(model, query: str):
    return model.encode(query)

def cosine_similiarity_func(query, base): 
    return cosine_similarity(query,base)

def search_top_k(scores):
    return 