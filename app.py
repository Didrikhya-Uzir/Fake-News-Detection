# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:59:11 2023

@author: didri
"""

import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    raw_content = [str(x) for x in request.form.values()]
    content = ' '.join(raw_content)
    content = stemming(content)
    content = [content]  # Wrap the string in a list to make it iterable
    # Use the same vectorizer for transforming
    content = vectorizer.transform(content)

    prediction = model.predict(content)
    
    if prediction == 1:
        prediction_text = "The news is Fake"
    elif prediction == 0:
        prediction_text = "The news is Real"
    else:
        prediction_text = "Invalid prediction"
    
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
    