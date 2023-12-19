import pickle
import numpy as np
from rank_bm25 import BM25Okapi
import re
from nltk.corpus import stopwords
import pandas as pd
from numpy.random import randint



if __name__ == "__main__":
    
    with open('id2ques.txt', 'rb') as f:
        id2ques = pickle.load(f)

    ques_ids = randint(1, max(id2ques.keys())+1, 5)
    ques_list = [id2ques[id] for id in ques_ids]
    api_call = {"questions": ques_list}

    print(api_call)
    