import os
import json
import numpy as np
import tensorflow as tf
import pickle
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS # Tambahkan ini untuk mengizinkan CORS

app = Flask(__name__)
CORS(app) # Mengaktifkan CORS untuk semua rute

# --- Konfigurasi Path Model ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'sentiment', 'sentiment_model_lstm.h5')
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'sentiment', 'tokenizer.pkl')
WORD_INDEX_PATH = os.path.join(os.path.dirname(__file__), 'models', 'sentiment', 'word_index.json')
LEXICON_POSITIVE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'sentiment', 'lexicon_positive.json')
LEXICON_NEGATIVE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'sentiment', 'lexicon_negative.json')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'search', 'tfidf_vectorizer.pkl')
MATRIX_PATH = os.path.join(os.path.dirname(__file__), 'models', 'search', 'tfidf_matrix.pkl')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'models', 'search', 'cbr_clean.json')

# --- Global Variables for ML Assets ---
sentiment_model = None
tokenizer = None
lexicon_positive = {}
lexicon_negative = {}
stemmer = None # Sastrawi stemmer
# --- Load Vectorizer for Search ---
tfidf_vectorizer = None
tfidf_matrix = None
data = None
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()


# --- Maximum Sequence Length for Padding (sesuaikan dengan model Anda) ---
MAX_SEQUENCE_LENGTH = 10

# --- Fungsi untuk Memuat Aset ML ---
def load_ml_assets():
    global sentiment_model, tokenizer, lexicon_positive, lexicon_negative, stemmer, tfidf_vectorizer, tfidf_matrix, data
    try:
        sentiment_model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Sentiment model loaded from: {MODEL_PATH}")

        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        print("Tokenizer loaded from tokenizer.pkl")

        with open(LEXICON_POSITIVE_PATH, 'r', encoding='utf-8') as f:
            lexicon_positive = json.load(f)
        print(f"Positive lexicon loaded from: {LEXICON_POSITIVE_PATH}")

        with open(LEXICON_NEGATIVE_PATH, 'r', encoding='utf-8') as f:
            lexicon_negative = json.load(f)
        print(f"Negative lexicon loaded from: {LEXICON_NEGATIVE_PATH}")

        # Muat vectorizer dan matriks dari file .pkl
        with open(VECTORIZER_PATH, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open(MATRIX_PATH, 'rb') as f:
            tfidf_matrix = pickle.load(f)
        print("Model TF-IDF berhasil dimuat.")
        # Muat data CBR
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("Data Pantai berhasil dimuat.")
        
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        print("Sastrawi Stemmer loaded.")

    except FileNotFoundError as e:
        print(f"Error: File model atau data tidak ditemukan. Pastikan file .pkl dan CSV.")
        print(e)
    except Exception as e:
        print(f"Error loading ML assets: {e}")
        sentiment_model = None

load_ml_assets()

# --- Fungsi Preprocessing ---
def load_slangwords(file_path):
    slangwords = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            slangwords = json.load(file)  # Load JSON data
    except Exception as e:
        print(f"Error loading slangwords: {e}")
    return slangwords

def load_stopwords(file_path):
    stopwords = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                stopwords.add(line.strip())
    except Exception as e:
        print(f"Error loading stopwords: {e}")
    return stopwords

slangwords = load_slangwords(os.path.join(os.path.dirname(__file__), 'models', 'sentiment', 'combined_slang_words.txt'))
indonesia_stopwords = load_stopwords(os.path.join(os.path.dirname(__file__), 'models', 'sentiment', 'combined_stop_words.txt'))
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

def tokenizing_text(text, tokenizer):
    sequences = tokenizer.texts_to_sequences([text])  # Mengubah teks menjadi token numerik
    return sequences[0]  # Mengembalikan daftar token

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

@app.route('/search-point', methods=['POST'])
def search_point():
    if tfidf_vectorizer is None or tfidf_matrix is None or data is None:
        return jsonify({"error": "Model atau data belum dimuat di server."}), 500

    data = request.get_json()
    user_input = data.get('query')
    top_n = 10  # Jumlah rekomendasi yang diinginkan

    if not user_input or not isinstance(user_input, str):
        return jsonify({"error": "Input pencarian tidak valid."}), 400

    try:
        # Opsional: Lakukan preprocessing pada input pengguna jika diperlukan
        user_input_processed = fix_slangwords(user_input.lower())  # Mengatasi slangwords
        tokenized = tokenizing_text(user_input_processed)  # Tokenisasi teks
        filtered = filtering_text(tokenized)  # Menghapus stopwords
        stemmed_text = stemming_text_func(' '.join(filtered))  # Stemming teks

        # Transformasi input pengguna menjadi vektor TF-IDF
        user_tfidf = tfidf_vectorizer.transform([user_input_processed])

        # Hitung kemiripan cosine
        cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

        # Ambil indeks top-N rekomendasi
        top_indices = cosine_sim.argsort()[-top_n:][::-1]

        # Siapkan hasil rekomendasi
        recommendations = []
        for i in top_indices:
            # Pastikan index valid sebelum mengakses data
            if 0 <= i < len(data):
                beach_info = data[i]
                # Tambahkan skor kemiripan ke hasil
                beach_info['similarity_score'] = float(cosine_sim[i])  # Konversi ke float biasa
                recommendations.append(beach_info)

        return jsonify({"recommendations": recommendations})

    except Exception as e:
        print("Error saat mencari rekomendasi:", e)
        return jsonify({"error": "Terjadi kesalahan saat memproses permintaan pencarian."}), 500
    
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