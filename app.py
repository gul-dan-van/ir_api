from flask import Flask, request, jsonify
import json
from retriever import BM25
import re
from nltk.corpus import stopwords
import pickle


with open('retrievers.txt', 'rb') as f:
    categorized_bm25 = pickle.load(f)
with open('ques2id.txt', 'rb') as f:
    ques2id = pickle.load(f)
with open('id2ques.txt', 'rb') as f:
    id2ques = pickle.load(f)
with open('ques2cat.txt', 'rb') as f:
    ques2cat = pickle.load(f)

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]


app = Flask(__name__)

def selection_func(query, min_selection=1):
    
    ques_list = []
    for ques in query:
        category = ques2cat[ques]
        bm25 = categorized_bm25[category]

        doc_scores = bm25.get_scores(ques)
        doc_scores = [
            (score, bm25.corpus[i]) for i, score in enumerate(doc_scores)
        ]

        ques_list += doc_scores
    
    sorted_score = sorted(ques_list)[::-1]
    final_list = [
        temp for _, temp in sorted_score
    ]
        
    return final_list[:max(1,min_selection)]


@app.route("/next-question-api/")
def get_user():
    data = {}

    # query = request.args.get("query")
    query = {'questions': ['Are tabletop exercises conducted regularly to test the effectiveness of the incident response plan and improve preparedness for ransomware incidents?', 'Are employees provided with regular training on the importance of data security, including recognizing and reporting potential ransomware threats?', ' Is sensitive data, both at rest and in transit, encrypted to mitigate the risk of unauthorized access in case of a security breach?', " Are simulated phishing exercises conducted to assess and improve employees' resilience to phishing attacks and ransomware threats?", ' Is there a plan in place for coordinating with law enforcement and other relevant external agencies in the event of a ransomware incident?']}
    if query:
        data["result"] = selection_func(query['questions'], 5)

    return jsonify(data), 200


if __name__ == "__main__":
    app.run(debug=True)

