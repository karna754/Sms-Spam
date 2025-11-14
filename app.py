from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
import string

def download_nltk_data():
    """Downloads required NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Download NLTK data
download_nltk_data()

# Load the pre-trained model and vectorizer
try:
    model = joblib.load('spam_detection_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
except:
    model = None
    vectorizer = None

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Join tokens back to text
    return ' '.join(tokens)

@app.route('/')
def home():
    return jsonify({"message": "SMS Spam Detection API is running!"})

@app.route('/detect', methods=['POST'])
def detect_spam():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        if not model or not vectorizer:
            return jsonify({
                "error": "Model not loaded. Please train the model first."
            }), 503
        
        # Preprocess the message
        processed_message = preprocess_text(message)
        
        # Vectorize the message
        message_vector = vectorizer.transform([processed_message])
        
        # Make prediction
        prediction = model.predict(message_vector)[0]
        probability = model.predict_proba(message_vector)[0][1]  # Probability of being spam
        
        return jsonify({
            "is_spam": bool(prediction),
            "probability": float(probability),
            "message": "SPAM detected!" if prediction else "This looks like a normal message."
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)