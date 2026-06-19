import streamlit as st
import fitz  # PyMuPDF
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="AI Learning Companion", layout="wide")
st.title("📚 AI Learning Companion (Smart Study Assistant)")

# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text(pdf_file):
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        text = ""
        for page in doc:
            text += page.get_text()

        return text

    except Exception as e:
        return f"Error reading PDF: {e}"

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------- SPLIT SENTENCES ----------------
def get_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

# ---------------- SMART SUMMARY ----------------
def generate_summary(sentences):
    return " ".join(sentences[:5])

# ---------------- KEYWORD MCQ GENERATOR ----------------
def generate_mcq(sentences):
    mcqs = []

    for s in sentences[:10]:
        words = s.split()
        if len(words) > 8:
            keyword = words[len(words)//2]  # simple keyword pick
            question = f"What is related to '{keyword}'?"
            mcqs.append((question, s))

    return mcqs[:5]

# ---------------- TF-IDF Q&A ----------------
def build_vectorizer(sentences):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences)
    return vectorizer, vectors

def answer_question(question, sentences, vectorizer, vectors):
    q_vec = vectorizer.transform([question])
    similarity = cosine_similarity(q_vec, vectors).flatten()

    idx = np.argmax(similarity)

    if similarity[idx] < 0.2:
        return "Answer not clearly found in document."

    return sentences[idx]

# ---------------- FLASHCARDS ----------------
def flashcards(sentences):
    cards = []
    for s in sentences[:8]:
        words = s.split()
        front = " ".join(words[:6]) + "..."
        cards.append({"front": front, "back": s})
    return cards

# ---------------- UI ----------------
uploaded_file = st.file_uploader("📄 Upload Study PDF", type=["pdf"])

if uploaded_file:

    raw_text = extract_text(uploaded_file)

    if "Error reading PDF" in raw_text:
        st.error(raw_text)

    else:
        text = clean_text(raw_text)
        sentences = get_sentences(text)

        # Vectorizer for Q&A
        vectorizer, vectors = build_vectorizer(sentences)

        st.subheader("📊 Document Overview")
        st.write("Total Sentences:", len(sentences))

        # ---------------- SUMMARY ----------------
        st.subheader("🧠 Smart Summary")
        summary = generate_summary(sentences)
        st.info(summary)

        # ---------------- MCQs ----------------
        st.subheader("🎯 Practice MCQs")
        mcqs = generate_mcq(sentences)

        for i, (q, _) in enumerate(mcqs):
            st.write(f"{i+1}. {q}")

        # ---------------- FLASHCARDS ----------------
        st.subheader("🧾 Flashcards")
        cards = flashcards(sentences)

        for c in cards:
            with st.expander(c["front"]):
                st.write(c["back"])

        # ---------------- Q&A ----------------
        st.subheader("💬 Ask Questions from PDF")

        user_q = st.text_input("Ask your doubt")

        if user_q:
            ans = answer_question(user_q, sentences, vectorizer, vectors)
            st.success(ans)

        # ---------------- DOWNLOAD REPORT ----------------
        report = f"""
AI STUDY REPORT
----------------

SUMMARY:
{summary}

TOTAL SENTENCES:
{len(sentences)}

MCQs:
{[q for q, _ in mcqs]}
"""

        st.download_button(
            "📥 Download Study Notes",
            report,
            file_name="study_report.txt"
        )
