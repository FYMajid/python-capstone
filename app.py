import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from flask_cors import CORS # Tambahkan ini untuk mengizinkan CORS

app = Flask(__name__)
CORS(app) # Mengaktifkan CORS untuk semua rute

# --- Konfigurasi Path Model ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'sentiment', 'dummy_sentiment_model.h5')
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'sentiment', 'tokenizer.json')
LEXICON_POSITIVE_PATH = os.path.join(os.path.dirname(__file__), 'model', 'sentiment', 'lexicon_positive.json')
LEXICON_NEGATIVE_PATH = os.path.join(os.path.dirname(__file__), 'model', 'sentiment', 'lexicon_negative.json')

# --- Global Variables for ML Assets ---
sentiment_model = None
tokenizer = None
lexicon_positive = {}
lexicon_negative = {}
stemmer = None # Sastrawi stemmer

# --- Maximum Sequence Length for Padding (sesuaikan dengan model Anda) ---
MAX_SEQUENCE_LENGTH = 10

# --- Fungsi untuk Memuat Aset ML ---
def load_ml_assets():
    global sentiment_model, tokenizer, lexicon_positive, lexicon_negative, stemmer
    try:
        sentiment_model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Sentiment model loaded from: {MODEL_PATH}")

        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
            tokenizer_json = f.read()
            tokenizer = tokenizer_from_json(tokenizer_json)
        print(f"Tokenizer loaded from: {TOKENIZER_PATH}")

        with open(LEXICON_POSITIVE_PATH, 'r', encoding='utf-8') as f:
            lexicon_positive = json.load(f)
        print(f"Positive lexicon loaded from: {LEXICON_POSITIVE_PATH}")

        with open(LEXICON_NEGATIVE_PATH, 'r', encoding='utf-8') as f:
            lexicon_negative = json.load(f)
        print(f"Negative lexicon loaded from: {LEXICON_NEGATIVE_PATH}")

        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        print("Sastrawi Stemmer loaded.")

    except Exception as e:
        print(f"Error loading ML assets: {e}")
        sentiment_model = None

load_ml_assets()

# --- Fungsi Preprocessing ---
slangwords = {
    "bgt": "banget", "gak": "tidak", "yg": "yang", "bgs": "bagus",
    "bgtu": "begitu", "kaga": "tidak", "tdk": "tidak", "jg": "juga", "udh": "sudah",
    
}

indonesia_stopwords = ["yang", "dan", "di", "ini", "itu", "tidak", "adalah", "untuk", "dengan", "dari", "ke", "pada"]
english_stopwords = ["the", "a", "an", "is", "of", "and", "to", "in", "for", "with", "on"]
custom_stopwords = ["iya", "yaa", "gak", "nya", "na", "sih", "ku", "di", "ga", "ya", "gaa", "loh", "kah", "woi", "woii", "woy", "banget", "oke"]
all_stopwords = set(list(indonesia_stopwords) + list(english_stopwords) + list(custom_stopwords))

def cleaning_text(text):
    text = str(text)
    text = text.replace(r'@[\w\d]+', ' ')
    text = text.replace(r'#[\w\d]+', ' ')
    text = text.replace(r'RT[\s]', ' ')
    text = text.replace(r'http\S+|www\S+', ' ')
    text = text.replace(r'\d+', ' ')
    text = text.replace(r'[^a-zA-Z\s]', ' ')
    text = text.replace('\n', ' ')
    text = text.replace(r'[^\w\s]', ' ')
    text = text.strip()
    return text

def casefolding_text(text):
    return text.lower()

def fix_slangwords(text):
    words = text.split()
    fixed_words = [slangwords.get(word, word) for word in words]
    return ' '.join(fixed_words)

def tokenizing_text(text):
    return text.split()

def filtering_text(words):
    return [word for word in words if word not in all_stopwords]

def stemming_text_func(text): 
    if stemmer:
        return stemmer.stem(text)
    return text


@app.route('/analyze-sentiment', methods=['POST'])
def analyze_sentiment():
    if sentiment_model is None or tokenizer is None:
        return jsonify({"error": "ML models not loaded or still initializing."}), 500

    data = request.get_json()
    review_text = data.get('review_text')

    if not review_text:
        return jsonify({"error": "No review_text provided"}), 400

    processed_text = cleaning_text(review_text)
    processed_text = casefolding_text(processed_text)
    processed_text = fix_slangwords(processed_text)
    tokenized = tokenizing_text(processed_text)
    filtered = filtering_text(tokenized)
    stemmed_text = stemming_text_func(' '.join(filtered))

    sequences = tokenizer.texts_to_sequences([stemmed_text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    prediction = sentiment_model.predict(padded_sequences)[0][0]
    sentiment_label = "Positive" if prediction >= 0.5 else "Negative"

    return jsonify({
        "sentiment": sentiment_label,
        "confidence": float(prediction)
    })

@app.route('/recommend-beach', methods=['POST'])
def recommend_beach():
    data = request.get_json()
    preference_text = data.get('preference_text')

    if not preference_text:
        return jsonify({"error": "No preference_text provided"}), 400

    

    
    dummy_recommendations = [
        {"placeId": "timur_0001", "score": 0.95}, 
        {"placeId": "barat_0005", "score": 0.88}, 
        {"placeId": "tengah_0010", "score": 0.76},
    ]
   
    return jsonify({"recommendations": dummy_recommendations})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 