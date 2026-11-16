
from flask import Flask, render_template, request
import pickle
import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)


# ==========================================
# 1. LOAD TRAINED MODEL AND VECTORIZER
# ==========================================

with open('spam_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    cv = pickle.load(file)

print("Model loaded:", type(model))
print("Vectorizer loaded:", type(cv))
print("Classes:", model.classes_)


# ==========================================
# 2. NLP PREPROCESSING
# ==========================================

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def preprocess_message(message):

    # Lowercase
    message = message.lower()

    # Remove URLs
    message = re.sub(r'http\S+|www\S+', '', message)

    # Remove punctuation
    message = message.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove numbers
    message = re.sub(r'\d+', '', message)

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
# 3. HOME PAGE
# ==========================================

@app.route('/')
def home():

    return render_template('index.html')


# ==========================================
# 4. SPAM DETECTION
# ==========================================
@app.route('/predict', methods=['POST'])
def predict():

    print("\n" + "=" * 50)
    print("          SMS SPAM DETECTION SYSTEM")
    print("=" * 50)

    # Get message
    message = request.form.get('message', '').strip()
    print("\nOriginal Message:")
    print(message)

    if not message:
        print("\nError: Empty message received.")
        return render_template(
            'index.html',
            prediction=None,
            message=''
        )

    # Preprocessing
    clean_message = preprocess_message(message)

    print("\nPreprocessed Message:")
    print(clean_message)

    # Convert into BoW
    message_vector = cv.transform([clean_message])

    print("\nVector Shape:", message_vector.shape)
    print("Non-zero Features:", message_vector.nnz)

    # Prediction
    prediction = model.predict(message_vector)[0]

    print("\n" + "-" * 50)

    if prediction == 'spam':

        result = 'spam'

        print("🚨 PREDICTION: SPAM MESSAGE")

    else:

        result = 'ham'

        print("✅ PREDICTION: NOT A SPAM MESSAGE")

    print("-" * 50)
    print("          ANALYSIS COMPLETED")
    print("=" * 50 + "\n")

    # Return result to website
    return render_template(
        'index.html',
        prediction=result,
        message=message
    )


    # Preprocess message
    clean_message = preprocess_message(message)

    print("Cleaned message:", clean_message)

    # Convert into Bag of Words
    message_vector = cv.transform([clean_message])

    print("Vector shape:", message_vector.shape)
    print("Non-zero features:", message_vector.nnz)

    # Predict using trained model
    prediction = model.predict(message_vector)[0]

    print("Model prediction:", prediction)

    # Send result to HTML
    if prediction == 'spam':

        result = 'spam'

    else:

        result = 'ham'

    print("Final result:", result)

    print("========== PREDICTION FINISHED ==========\n")

    return render_template(
        'index.html',
        prediction=result,
        message=message
    )


# ==========================================
# 5. RUN FLASK APPLICATION
# ==========================================

if __name__ == '__main__':

    app.run(debug=True)
