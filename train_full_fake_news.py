# -----------------------------------------
# ADVANCED FAKE NEWS DETECTION SYSTEM
# WITH TF-IDF, NLTK PREPROCESSING,
# NAIVE BAYES, RF, GB (SAFE), STACKING
# -----------------------------------------

import os
import re
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- SKLEARN ---
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

# --- NLTK ---
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# -----------------------------------------
# CLEANING FUNCTION
# -----------------------------------------
def clean_text_advanced(text):
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


# -----------------------------------------
# LOAD DATA
# -----------------------------------------
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

df_fake["label"] = "FAKE"
df_true["label"] = "REAL"

df = pd.concat([df_fake, df_true], ignore_index=True)
df["combined"] = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)

print("Cleaning text...")
df["clean_text"] = df["combined"].apply(clean_text_advanced)

# Encode labels
le = LabelEncoder()
df["y"] = le.fit_transform(df["label"])

X = df["clean_text"]
y = df["y"]

# -----------------------------------------
# TRAIN-TEST SPLIT
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.18, random_state=42
)

# -----------------------------------------
# TF-IDF (15000 FEATURES)
# -----------------------------------------
tfidf = TfidfVectorizer(
    max_features=15000,       # SAFE + STRONG
    min_df=3,
    ngram_range=(1, 2),
    stop_words="english"
)

print("Vectorizing...")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# -----------------------------------------
# CLASS WEIGHTS
# -----------------------------------------
cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
cw = {i: cw[i] for i in range(len(cw))}
print("Class Weights:", cw)


# -----------------------------------------
# NAIVE BAYES + GRIDSEARCH
# -----------------------------------------
nb = MultinomialNB()
nb_params = {"alpha":[0.1,0.5,1,2],"fit_prior":[True,False]}

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

g_nb = GridSearchCV(nb, nb_params, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
g_nb.fit(X_train_tfidf, y_train)
best_nb = g_nb.best_estimator_

pred_nb = best_nb.predict(X_test_tfidf)
print("\n=== NB RESULTS ===")
print("Accuracy:", accuracy_score(y_test,pred_nb))
print(classification_report(y_test,pred_nb))


# -----------------------------------------
# DECISION TREE
# -----------------------------------------
dt = DecisionTreeClassifier(max_depth=40, class_weight="balanced", random_state=42)
dt.fit(X_train_tfidf, y_train)
pred_dt = dt.predict(X_test_tfidf)

print("\n=== DECISION TREE RESULTS ===")
print("Accuracy:", accuracy_score(y_test,pred_dt))
print(classification_report(y_test,pred_dt))


# -----------------------------------------
# RANDOM FOREST
# -----------------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=40,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_tfidf, y_train)
pred_rf = rf.predict(X_test_tfidf)

print("\n=== RANDOM FOREST RESULTS ===")
print("Accuracy:", accuracy_score(y_test,pred_rf))
print(classification_report(y_test,pred_rf))


# -----------------------------------------
# GRADIENT BOOSTING (SAFE + EFFECTIVE)
# -----------------------------------------
print("Preparing dense matrix...")
X_train_dense = X_train_tfidf.toarray().astype("float16")
X_test_dense  = X_test_tfidf.toarray().astype("float16")

gb = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.07,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

print("Training Gradient Boosting...")
gb.fit(X_train_dense, y_train)
pred_gb = gb.predict(X_test_dense)

print("\n=== GRADIENT BOOSTING RESULTS ===")
print("Accuracy:", accuracy_score(y_test,pred_gb))
print(classification_report(y_test,pred_gb))


# -----------------------------------------
# STACKING ENSEMBLE MODEL
# -----------------------------------------
stack = StackingClassifier(
    estimators=[
        ("nb", best_nb),
        ("rf", rf),
        ("gb", gb),
        ("dt", dt),
    ],
    final_estimator=LogisticRegression(max_iter=1200),
    n_jobs=-1
)

print("Training Stacking Model...")
stack.fit(X_train_tfidf, y_train)
pred_stack = stack.predict(X_test_tfidf)

print("\n=== STACKING MODEL RESULTS ===")
print("Accuracy:", accuracy_score(y_test,pred_stack))
print(classification_report(y_test,pred_stack))


# -----------------------------------------
# SAVE MODELS
# -----------------------------------------
os.makedirs("artifacts", exist_ok=True)

joblib.dump(tfidf, "artifacts/tfidf.joblib")
joblib.dump(best_nb, "artifacts/nb.joblib")
joblib.dump(dt, "artifacts/dt.joblib")
joblib.dump(rf, "artifacts/rf.joblib")
joblib.dump(gb, "artifacts/gb.joblib")
joblib.dump(stack, "artifacts/stack.joblib")

print("\nAll models saved! Training complete!")
