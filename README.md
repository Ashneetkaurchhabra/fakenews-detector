# fakenews-detector

# 📰 Fake News Detector — Local Usage Guide

This project is a full **Fake News Detection System** that runs completely on your computer.

It includes:

- `train_full_fake_new.py` → trains ML models  
- `Fake.csv` + `True.csv` → dataset files **kept in the root folder**  
- `artifacts/` → generated models  
- `app.py` → backend API  
- `frontend/` → HTML/CSS/JS user interface  

This guide explains how to use the project from scratch.

---

# 📁 Project Structure

Make sure your folder looks like this:

```
fake-news-detector/
│
├── train_full_fake_new.py
├── app.py
├── requirements.txt
│
├── Fake.csv
├── True.csv
│
├── artifacts/              ← AUTO-GENERATED after training
│     ├── tfidf.joblib
│     ├── nb.joblib
│     ├── dt.joblib
│     ├── rf.joblib
│     ├── gb.joblib
│     └── stack.joblib
│
└── frontend/
       ├── index.html
       ├── style.css
       ├── script.js
       └── bg1.png
```

✔ CSV files are **in the main root**, EXACTLY like you wanted.  
❌ No separate data folder.  
❌ No backend folder.  
❌ Only nice and simple.

---

# 📥 1. Download Dataset (CSV Files)

This project uses the **Fake and Real News Dataset**:

🔗 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Download and extract the dataset.

You will get:

- `Fake.csv`  
- `True.csv`

Place BOTH files directly in the **project root folder**, like this:

```
fake-news-detector/
   Fake.csv
   True.csv
   train_full_fake_new.py
   app.py
   ...
```

NOT in a separate folder.

---

# 🧰 2. Install Requirements

### Open VS Code
- File → Open Folder → select the project folder

### Open Terminal
Terminal → New Terminal

### Create Virtual Environment

#### Windows:
```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Install NLTK data
```bash
python - <<'PY'
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
PY
```

---

# 🤖 3. TRAIN THE MODEL (IMPORTANT)

Run the training script:

```bash
python train_full_fake_new.py
```

This will:

✔ Read `Fake.csv` and `True.csv` from the root  
✔ Clean the text  
✔ Train multiple ML models  
✔ Create an `artifacts/` folder  
✔ Save all `.joblib` files inside it  

After training finishes, you MUST see:

```
artifacts/
   tfidf.joblib
   nb.joblib
   dt.joblib
   rf.joblib
   gb.joblib
   stack.joblib
```

If this folder exists → training succeeded.

---

# 🌐 4. Run Backend (API)

Your backend is:

```
app.py
```

Run it:

```bash
python app.py
```

It starts here:

```
http://127.0.0.1:5000
```

### Test it:

```bash
curl -X POST -H "Content-Type: application/json" \
-d "{\"text\":\"this is a test news article\"}" \
http://127.0.0.1:5000/predict
```

You should receive JSON output.

---

# 🎨 5. Run Frontend

Your UI is inside:

```
frontend/
   index.html
   style.css
   script.js
   bg1.png
```

### Easiest method:
Open `index.html` in your browser.

OR run a local server:

```bash
cd frontend
python -m http.server 8000
```

Open:

```
http://localhost:8000
```

Paste your news → click **Analyze** → your backend will respond.

Backend **must** be running for the frontend to work.

---


alsooo lastlyyyy install all the libraries from requirements.txt
