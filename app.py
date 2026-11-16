
from flask import Flask, render_template, request
import pickle
import re
import string
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# =====================================
# LOAD MODEL AND VECTORIZER
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'spam_model.pkl'), 'rb') as file:
    model = pickle.load(file)

with open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb') as file:
    cv = pickle.load(file)

print("Model loaded:", type(model))
print("Vectorizer loaded:", type(cv))
print("Classes:", model.classes_)

# =====================================
# NLP PREPROCESSING
# =====================================

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def preprocess_message(message):

    message = message.lower()

    message = re.sub(r'http\S+|www\S+', '', message)

    message = message.translate(
        str.maketrans('', '', string.punctuation)
    )

    message = re.sub(r'\d+', '', message)

    message = ' '.join(message.split())

    words = word_tokenize(message)

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    stemmed_words = [
        ps.stem(word) for word in filtered_words
    ]

    return ' '.join(stemmed_words)


# =====================================
# HOME PAGE
# =====================================

@app.route('/')
def home():
    return render_template('index.html')


# =====================================
# SPAM PREDICTION
# =====================================

@app.route('/predict', methods=['POST'])
def predict():

    message = request.form.get('message', '').strip()

    if not message:
        return render_template(
            'index.html',
            prediction=None,
            message=''
        )

    clean_message = preprocess_message(message)

    message_vector = cv.transform([clean_message])

    prediction = model.predict(message_vector)[0]

    if prediction == 'spam':
        result = 'spam'
    else:
        result = 'ham'

    print("Original:", message)
    print("Cleaned:", clean_message)
    print("Prediction:", result)

    return render_template(
        'index.html',
        prediction=result,
        message=message
    )


# =====================================
# RUN APPLICATION
# =====================================

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False
    )
