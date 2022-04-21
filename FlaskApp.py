# -*- coding: utf-8 -*-
import numpy as np
from keras.models import load_model
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Retrieve the saved sentiment analysis model
model = load_model('model-2_layer_32-lr_001.h5')
# Retrieve the saved tokenizer from file
token_obj = pickle.load(open('tokenizer.pkl', 'rb'))

# Define the maximum sequence length
MAX_LENGTH = 50
# Find the vocabulary size
vocab_size = len(token_obj.word_index) + 1

# Function for predicting the sentiment of a review
def predict_sentiment(sentence):
    # Convert the sentence to lowercase, and put it in a list for tokenizing
    sentence = [sentence.lower()]
    # Tokenize the sentence using the saved tokenizer
    tokens = token_obj.texts_to_sequences(sentence)
    # Pad the sequence to be max length
    padded = pad_sequences(tokens, maxlen = MAX_LENGTH, padding = 'post')
    # Get the score predicted by the model, 0 - rotten, 1 - fresh
    score = model.predict(padded)[0][0]
    # If the model predicted the score to be greater than or equal to 0.5,
    # meaning closer to fresh
    if score >= 0.5:
        # Set the resulting label as FRESH
        result = 'FRESH'
        # Set the colour to the green used by rotten tomatoes
        colour = '#FF0000'
    # Otherwise, meaning the model predict the score to be closer to rotten
    else:
        # Set the resulting label as ROTTEN
        result = 'ROTTEN'
        # Set the colour to the red used by rotten tomatoes
        colour = '#3d5e2f'
    # Round the score to 5 decimal places for easier viewing
    score = round(score, 5)
    return result, score, colour

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    sentence = [x for x in request.form.values()][0]
    result, score, colour = predict_sentiment(sentence)
    return render_template('prediction.html', sentence = sentence,
                            result = result, score = score, colour = colour)

if __name__ == '__main__':
    app.run()