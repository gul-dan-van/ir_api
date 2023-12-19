import pickle
import numpy as np
from rank_bm25 import BM25Okapi
import re
from nltk.corpus import stopwords
import pandas as pd


stop_words = set(stopwords.words('english'))

class BM25:

    def __init__(self, corpus):
        self.corpus = corpus
        tokenized_corpus = [self.preprocess_text(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = [word for word in text.split() if word not in stop_words]
    
        return tokens

    def get_scores(self, query):
        tokenized_query = self.preprocess_text(query)
        doc_scores = self.bm25.get_scores(tokenized_query)

        return doc_scores


if __name__ == "__main__":
    df = pd.read_csv('Questionnaire DataSet - Sheet1.csv')
    df = df.drop(df.index[df["ID"].isna()])
    df = df.drop(["ID"], axis=1)
    df.index = range(1, df.shape[0]+1)
    
    grouped_df = df.groupby('CATEGORY')
    
    categories_list = df['CATEGORY'].unique().tolist()
    categorized_bm25 = {}

    for category in categories_list:
        temp_df = grouped_df.get_group(category)
        bm25 = BM25(temp_df['QUESTION'].tolist())
        categorized_bm25[category] = bm25

    with open('retrievers.txt', 'wb') as f:
        pickle.dump(categorized_bm25, f)

    ques_id = {
        ques: id for ques, id in zip(df['QUESTION'], df.index)
    }
    with open('ques2id.txt', 'wb') as f:
        pickle.dump(ques_id, f)
    
    id_ques = {
        id: ques for ques, id in zip(df['QUESTION'], df.index)
    }
    with open('id2ques.txt', 'wb') as f:
        pickle.dump(id_ques, f)
    
    ques_category = {
        ques: category for ques, category in zip(df['QUESTION'], df['CATEGORY'])
    }
    with open('ques2cat.txt', 'wb') as f:
        pickle.dump(ques_category, f)
