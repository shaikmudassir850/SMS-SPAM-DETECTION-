
from flask import Flask, render_template, request
import pickle
import re
import string
import os
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


app = Flask(__name__)


# ==========================================
# 1. PROJECT PATH
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# NLTK data directory
NLTK_DATA_DIR = os.path.join(BASE_DIR, 'nltk_data')

os.makedirs(NLTK_DATA_DIR, exist_ok=True)

nltk.data.path.insert(0, NLTK_DATA_DIR)


# ==========================================
# 2. DOWNLOAD NLTK DATA
# ==========================================

print("Downloading NLTK resources...")

resources = ['stopwords', 'punkt', 'punkt_tab']

for resource in resources:

    success = nltk.download(
        resource,
        download_dir=NLTK_DATA_DIR,
        quiet=True
    )

    if not success:
        raise RuntimeError(
            f"Failed to download NLTK resource: {resource}"
        )

print("NLTK resources downloaded successfully!")


# ==========================================
# 3. LOAD TRAINED MODEL AND VECTORIZER
# ==========================================

with open(
    os.path.join(BASE_DIR, 'spam_model.pkl'), 'rb'
) as file:
    model = pickle.load(file)


with open(
    os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb'
) as file:
    cv = pickle.load(file)


print("Model loaded:", type(model))
print("Vectorizer loaded:", type(cv))
print("Classes:", model.classes_)


# ==========================================
# 4. NLP PREPROCESSING
# ==========================================

stop_words = set(stopwords.words('english'))

ps = PorterStemmer()


def preprocess_message(message):

    # Lowercase
    message = message.lower()

    # Remove URLs
    message = re.sub(
        r'http\S+|www\S+',
        '',
        message
    )

    # Remove punctuation
    message = message.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove numbers
    message = re.sub(r'\d+', '', message)

    # Normalize whitespace
    message = ' '.join(message.split())

    # Tokenization
    words = word_tokenize(message)

    # Remove stopwords
    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    # Stemming
    stemmed_words = [
        ps.stem(word) for word in filtered_words
    ]

    # Join words
    return ' '.join(stemmed_words)


# ==========================================
# 5. HOME PAGE
# ==========================================

@app.route('/')
def home():

    return render_template('index.html')


# ==========================================
# 6. SPAM DETECTION
# ==========================================

@app.route('/predict', methods=['POST'])
def predict():

    print("\n========== SMS SPAM DETECTION ==========")

    # Get message
    message = request.form.get(
        'message', ''
    ).strip()

    if not message:

        return render_template(
            'index.html',
            prediction=None,
            message=''
        )

    # Preprocessing
    clean_message = preprocess_message(message)

    print("Original Message:", message)
    print("Cleaned Message:", clean_message)

    # Convert into Bag of Words
    message_vector = cv.transform([clean_message])

    print("Vector Shape:", message_vector.shape)

    # Prediction
    prediction = model.predict(message_vector)[0]

    if prediction == 'spam':

        result = 'spam'

        print("PREDICTION: SPAM MESSAGE")

    else:

        result = 'ham'

        print("PREDICTION: NOT SPAM")

    print("========================================\n")

    # Return result to website
    return render_template(
        'index.html',
        prediction=result,
        message=message
    )


# ==========================================
# 7. RUN FLASK APPLICATION
# ==========================================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False
    )
