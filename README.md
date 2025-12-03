# 📄 AI PDF Chat Assistant (RAG)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Groq API](https://img.shields.io/badge/API-Groq-orange)](https://groq.com/)

A lightning-fast **Retrieval-Augmented Generation (RAG)** application allowing users to chat with their PDF documents.  
Powered by **Groq (Llama 3)** for instant inference, **FAISS** for vector search, and **Streamlit** for the interface.

---

## 🚀 Features

- **📄 PDF Upload:** Seamlessly extract text from PDF files using `pdfplumber`.
- **⚡ Ultra-Fast Responses:** Uses **Groq API** (Llama 3.1 8B) for near-instant answers.
- **🔍 Vector Search:** Efficient context retrieval using **FAISS** and **HuggingFace Embeddings**.
- **🛡️ Secure:** API keys are managed safely via Streamlit Secrets (not hardcoded).
- **💬 Conversational UI:** Clean and simple chat interface.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM:** Llama 3.1-8b (via Groq API)
- **Embeddings:** `all-MiniLM-L6-v2` (Sentence Transformers)
- **Vector DB:** FAISS (Facebook AI Similarity Search)
- **PDF Processing:** pdfplumber

---

## 💻 Local Installation

Follow these steps to run the app on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/repo-name.git
cd repo-name
```

---

## 2. Create a Virtual Environment (Recommended)
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

---

## 3. Install Dependencies
pip install -r requirements.txt

---

## 4. Set up API Keys 🔑
Important: This project uses secrets.toml for security.

1. Create a folder named .streamlit in the root directory.
2. Inside it, create a file named secrets.toml.
3. Add your Groq API key:

# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

---

# 5. Run the App
streamlit run app.py

---

## ☁️ Deployment on Streamlit Cloud
To deploy this app online while keeping your API key secure:

1. Push your code to GitHub (Ensure .streamlit/secrets.toml is in your .gitignore and NOT uploaded).
2. Go to Streamlit Community Cloud.
3. Connect your GitHub repo and deploy the app.
4. The app will fail initially (missing API Key). Don't panic!
5. Go to your App Settings -> Secrets.
6. Paste your API key configuration there:
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

Reboot the app, and it will work perfectly! 🚀

---

## 📂 Project Structure
```
├── .streamlit/
│   └── secrets.toml      # API Keys (Local only - DO NOT UPLOAD)
├── app.py                # Main application logic
├── requirements.txt      # Python dependencies
├── .gitignore            # Files to ignore (secrets, venv, etc.)
└── README.md             # Project documentation
```
