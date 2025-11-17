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
       └── bg.png
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
   bg.png
```

### Easiest method:
Open `index.html` in your browser.

OR Run the Frontend (Using Live Server)

Your frontend files are inside:

```
frontend/
   index.html
   style.css
   script.js
   bg.png
```

To view the website in your browser, follow these **very simple steps**:

---

### ✅ Step 1 — Open the Project in VS Code
1. Open **Visual Studio Code**
2. Click **File → Open Folder**
3. Select your project folder:
   ```
   fake-news-detector/
   ```

---

### ✅ Step 2 — Install the Live Server Extension
1. Click the **Extensions** icon on the left sidebar (looks like 4 squares)
2. In the top search bar, type:
   ```
   Live Server
   ```
3. Install the extension named **Live Server (by Ritwick Dey)**

This enables one-click running of websites.

---

### ✅ Step 3 — Open the Frontend Folder
In the VS Code Explorer (left side), open:

```
frontend/index.html
```

---

### ✅ Step 4 — Right-Click on `index.html`
Right-click → **Open with Live Server**

OR click the **Go Live** button in the bottom-right corner of VS Code.

---

### ✅ Step 5 — Website Opens Automatically
Your browser will open a link like:

```
http://127.0.0.1:5500/frontend/index.html
```

This is your **live frontend**, running locally.

Every time you press **CTRL + S**, Live Server automatically refreshes the page.

---

### ❗ IMPORTANT: Backend Must Be Running
Live Server only runs your **frontend**.

To get predictions, the backend must also be running in another terminal:

```
python app.py
```

Backend runs on:

```
http://127.0.0.1:5000
```

Frontend (Live Server) runs on:

```
http://127.0.0.1:5500
```

Both work together normally.

---

### ⭐ If Live Server Does NOT Open Automatically
Do this:

1. Press **CTRL + SHIFT + P**
2. Type:
   ```
   Live Server: Open with Live Server
   ```
3. Press Enter

It will open your site.

---

### 🎉 Done!
Your frontend is now running perfectly with Live Server.


Paste your news → click **Analyze** → your backend will respond.

Backend **must** be running for the frontend to work.
