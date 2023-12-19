from flask import Flask, request, jsonify
from retriever import BM25
import re
from nltk.corpus import stopwords
import pickle


with open('ques2id.txt', 'rb') as f:
    ques2id = pickle.load(f)

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]

bm25 = BM25(corpus)

app = Flask(__name__)

def selection_func(bm25, query, min_selection=1):
        
    if type(query) == str:
        doc_scores = bm25.get_scores(query)
        doc_scores = sorted([(score, i) for i, score in enumerate(doc_scores)])[::-1]
        
        return [bm25.corpus[i] for _, i in doc_scores[:min(1, min_selection)]]
    
    elif type(query) == list:
        ques_list = []
        for ques in query:
            doc_scores = bm25.get_scores(ques)
            ques_list.append([(score, bm25.corpus[i]) for i, score in enumerate(doc_scores)])


@app.route("/get-user/")
def get_user():
    data = {}

    query = request.args.get("query")
    ques_list = query.split('+'*5)
    if query:
        data["result"] = selection_func(bm25, query, 1)

    return jsonify(data), 200


if __name__ == "__main__":
    app.run(debug=True)


