from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
import nltk

from flask_cors import CORS   # CORS import

# ---------------------------------------------
# INITIALIZE FLASK + ENABLE FULL CORS
# ---------------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Load NLTK
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# ---------------------------------------------
# CLEANING FUNCTION
# ---------------------------------------------
def clean_text(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOPWORDS]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

# ---------------------------------------------
# LOAD MODELS
# ---------------------------------------------
tfidf = joblib.load("artifacts/tfidf.joblib")
nb_model = joblib.load("artifacts/nb.joblib")
dt_model = joblib.load("artifacts/dt.joblib")
rf_model = joblib.load("artifacts/rf.joblib")
gb_model = joblib.load("artifacts/gb.joblib")
stack_model = joblib.load("artifacts/stack.joblib")


def decode(pred):
    return "REAL" if pred == 1 else "FAKE"


# ---------------------------------------------
# PREDICT ROUTE
# ---------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "No 'text' provided"}), 400

    raw_text = data["text"]
    cleaned = clean_text(raw_text)

    X = tfidf.transform([cleaned])

    pred_nb = decode(nb_model.predict(X)[0])
    pred_dt = decode(dt_model.predict(X)[0])
    pred_rf = decode(rf_model.predict(X)[0])
    pred_gb = decode(gb_model.predict(X.toarray().astype("float16"))[0])
    pred_stack = decode(stack_model.predict(X)[0])

    return jsonify({
        "Naive Bayes": pred_nb,
        "Decision Tree": pred_dt,
        "Random Forest": pred_rf,
        "Gradient Boosting": pred_gb,
        "Stacking Model": pred_stack,
        "Final Verdict": pred_stack
    })


# ---------------------------------------------
# RUN SERVER
# ---------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)

