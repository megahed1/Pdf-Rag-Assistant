import streamlit as st
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# --------------------------
# 1. إعداد النموذج والـ API
# --------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except FileNotFoundError:
    st.error("ملف الأسرار secrets.toml مش موجود! تأكد من إنشائه داخل مجلد .streamlit")
    st.stop()
except KeyError:
    st.error("المفتاح GROQ_API_KEY مش موجود داخل ملف secrets.toml")
    st.stop()
client = Groq(api_key=GROQ_API_KEY)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------
# 2. استخراج النص من PDF
# --------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


# --------------------------
# 3. إنشاء Embeddings
# --------------------------
def create_embeddings(chunks):
    vectors = embed_model.encode(chunks)
    vectors = np.array(vectors).astype("float32")
    return vectors


# --------------------------
# 4. بناء قاعدة البيانات FAISS
# --------------------------
def build_faiss_index(vectors):
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index


# --------------------------
# 5. RAG — استرجاع أقرب أجزاء من النص
# --------------------------
def search(query, chunks, index, k=3):
    query_vec = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)
    retrieved = "\n".join([chunks[i] for i in indices[0]])
    return retrieved


# --------------------------
# 6. Groq LLM — الإجابة
# --------------------------
def ask_groq(question, context):
    prompt = f"""
You are a helpful AI assistant. Answer the question based on the context.

Context:
{context}

Question: {question}

Answer:
    """
    chat_completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return chat_completion.choices[0].message.content


# --------------------------
# 7. واجهة Streamlit
# --------------------------
st.title("📄 PDF RAG Assistant with Groq")
st.write("ارفع PDF واسأل أي سؤال عنه!")

pdf_file = st.file_uploader("ارفع PDF", type=["pdf"])

if pdf_file:
    st.success("تم رفع الملف بنجاح ✔")

    st.write("📌 جاري استخراج النص...")
    text = extract_text_from_pdf(pdf_file)

    # تقسيم النص إلى أجزاء
    chunks = text.split("\n")
    chunks = [c.strip() for c in chunks if len(c.strip()) > 10]

    st.write("📌 جاري إنشاء Embeddings...")
    vectors = create_embeddings(chunks)

    st.write("📌 جاري إنشاء قاعدة بيانات FAISS...")
    index = build_faiss_index(vectors)

    question = st.text_input("❓ اسأل سؤال من الملف:")

    if st.button("إرسال"):
        with st.spinner("جارٍ البحث والإجابة..."):
            context = search(question, chunks, index)
            answer = ask_groq(question, context)

        st.subheader("🧠 الإجابة:")
        st.write(answer)
