# 📱 SMS Spam Detection System | Machine Learning & NLP

<p align="center">

  <strong>An intelligent SMS classification system powered by Natural Language Processing and Machine Learning.</strong>

  <br><br>

  <a href="https://sms-spam-detection-5i3j.onrender.com">
    <img src="https://img.shields.io/badge/Live%20Demo-Visit%20Application-8A2BE2?style=for-the-badge&logo=render&logoColor=white">
  </a>

  <img src="https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python&logoColor=white">

  <img src="https://img.shields.io/badge/Machine%20Learning-Naive%20Bayes-orange?style=for-the-badge">

  <img src="https://img.shields.io/badge/Framework-Flask-black?style=for-the-badge&logo=flask">

</p>

---

## 🚀 Live Application

**Try the application here:**

🔗 https://sms-spam-detection-5i3j.onrender.com

The application allows users to enter an SMS message and instantly classify it as:

- 🚨 **SPAM** – Unwanted or suspicious message.
- ✅ **NOT SPAM (HAM)** – Message classified as legitimate.

---

## 📌 Project Overview

SMS spam messages have become a common problem in digital communication. Fraudulent messages, unwanted promotions, fake rewards, and suspicious links can mislead users.

This project aims to solve this problem using Machine Learning and Natural Language Processing (NLP).

The SMS Spam Detection System analyzes the text entered by the user, processes the message using NLP techniques, converts it into numerical features using Bag of Words, and predicts whether the message is Spam or Ham.

The system uses a trained **Multinomial Naive Bayes classifier** to perform binary text classification.

---

## 🎯 Project Objectives

- Develop an intelligent SMS spam classification system.
- Apply NLP techniques to clean and process text data.
- Convert textual messages into numerical representations.
- Train a supervised Machine Learning classification model.
- Build an interactive web application using Flask.
- Deploy the application on Render.
- Enable real-time SMS classification through a web interface.

---

## ✨ Key Features

- **Real-Time Prediction:** Classifies user-entered SMS messages.
- **Natural Language Processing:** Performs tokenization, stopword removal, and stemming.
- **Text Cleaning:** Removes URLs, punctuation, and numerical characters.
- **Bag of Words:** Converts processed text into numerical features.
- **Naive Bayes Classification:** Uses MultinomialNB for Spam/Ham prediction.
- **Interactive Web Interface:** Provides a simple interface for message submission and displaying predictions.
- **Cloud Deployment:** Hosted on Render.

---

## 🧠 Machine Learning Workflow

The project follows an end-to-end supervised Machine Learning workflow.

### 1. Data Preprocessing

The raw SMS messages are cleaned and prepared for model training.

Techniques used:

- Lowercase conversion
- URL removal
- Punctuation removal
- Number removal
- Tokenization using NLTK
- Stopword removal
- Stemming using PorterStemmer
- Joining processed tokens into text

### 2. Feature Extraction

The project uses **Bag of Words (BoW)** with Scikit-learn's `CountVectorizer`.

It converts text messages into numerical feature vectors.

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)
```

### 3. Model Training

The classification algorithm used is:

**Multinomial Naive Bayes**

Multinomial Naive Bayes is suitable for text classification tasks involving word-frequency features.

The trained model and vectorizer are saved using Pickle for later use in the Flask application.

### 4. Prediction

When a user submits a new SMS:

1. The Flask application receives the message.
2. The message goes through NLP preprocessing.
3. The processed message is transformed using the saved CountVectorizer.
4. The trained Multinomial Naive Bayes model predicts the class.
5. The result is displayed on the website.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Classification Algorithm | Multinomial Naive Bayes |
| NLP | NLTK |
| Feature Extraction | CountVectorizer (Bag of Words) |
| Data Processing | Pandas, NumPy |
| Web Framework | Flask |
| Model Serialization | Pickle |
| Production Server | Gunicorn |
| Deployment Platform | Render |
| Development Environment | VS Code, Jupyter Notebook |

---

## 📂 Project Structure

```text
SMS_Spam_Detection/
│
├── app.py
│
├── requirements.txt
│
├── spam_model.pkl
│
├── vectorizer.pkl
│
├── notebook.ipynb
│
└── templates/
    └── index.html
```

### File Description

| File | Description |
|---|---|
| `app.py` | Flask backend and prediction functionality |
| `requirements.txt` | Python dependencies |
| `spam_model.pkl` | Trained Multinomial Naive Bayes model |
| `vectorizer.pkl` | Saved CountVectorizer |
| `notebook.ipynb` | Data preprocessing, model training and experimentation |
| `index.html` | Frontend interface using HTML and CSS |

---

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### Prerequisites

- Python 3.13.9
- Git
- VS Code (recommended)

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

```bash
cd SMS_Spam_Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Resources

```bash
python -m nltk.downloader stopwords punkt punkt_tab
```

### 5. Run the Flask Application

```bash
python app.py
```

Open your browser:

```text
http://127.0.0.1:5000/
```

---

## ☁️ Deployment

The application is deployed using **Render**.

Deployment configuration:

**Build Command:**

```bash
pip install -r requirements.txt && python -m nltk.downloader -d ./nltk_data stopwords punkt punkt_tab
```

**Start Command:**

```bash
gunicorn --bind 0.0.0.0:$PORT app:app
```

The trained model and vectorizer are loaded when the Flask application starts.

---

## 🔍 Example Predictions

The application accepts SMS messages and returns one of two classifications.

| Input Message | Expected Classification |
|---|---|
| Congratulations! You have won a free prize. | SPAM |
| Hey, are we meeting today? | NOT SPAM |
| Claim your reward now by clicking this link. | SPAM |
| I'll call you when I reach home. | NOT SPAM |

*Examples are illustrative; actual predictions depend on the trained model.*

---

## 📊 Model Information

| Component | Implementation |
|---|---|
| Learning Type | Supervised Learning |
| Problem Type | Binary Classification |
| Algorithm | Multinomial Naive Bayes |
| Feature Extraction | Bag of Words |
| Vectorizer | CountVectorizer |
| Text Processing | NLTK |
| Model Storage | Pickle |

---

## 🔮 Future Improvements

Some potential improvements for future versions:

- Compare additional Machine Learning algorithms.
- Improve model generalization on unseen SMS messages.
- Add prediction confidence and probability visualization.
- Introduce a larger and more diverse SMS dataset.
- Add an SMS message history feature.
- Improve multilingual spam detection.
- Add automated model evaluation and testing.

---

## ⚠️ Limitations

- The model's predictions depend on the training dataset.
- Unseen message patterns may result in incorrect classifications.
- Bag of Words does not understand the complete semantic meaning of a sentence.
- A message classified as NOT SPAM is not guaranteed to be safe.

This application is intended as a Machine Learning demonstration and should not be treated as a complete security or fraud-detection solution.

---

## 👨‍💻 Author

**Shaik Mudassir**

B.Tech – Computer Science and Engineering (AI & ML)

Interested in Machine Learning, Natural Language Processing, Artificial Intelligence, and Generative AI.

---

## ⭐ Support

If you find this project interesting, consider giving the repository a ⭐ on GitHub.

**Thank you for visiting!**
