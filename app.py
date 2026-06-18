import streamlit as st
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="AI Learning Companion",
    layout="wide"
)

st.title("📚 AI Learning Companion (Smart Study Assistant)")

# ---------------- PDF PARSER ----------------
def extract_text(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
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

# ---------------- SUMMARY ----------------
def generate_summary(text):
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return ". ".join(sentences[:5])

# ---------------- MCQ GENERATOR ----------------
def generate_mcq(text):
    sentences = text.split(".")
    questions = []

    for s in sentences:
        s = s.strip()
        if len(s) > 40:
            q = f"What is mentioned in: '{s[:50]}...?'"
            questions.append(q)

    return questions[:5]

# ---------------- FLASHCARDS ----------------
def flashcards(text):
    sentences = text.split(".")
    cards = []

    for s in sentences:
        s = s.strip()
        if len(s) > 40:
            cards.append({
                "front": s[:60] + "...",
                "back": s
            })

    return cards[:8]

# ---------------- SIMPLE Q&A ----------------
def answer_question(text, question):
    sentences = text.split(".")
    question_words = question.lower().split()

    for s in sentences:
        if any(word in s.lower() for word in question_words):
            return s.strip()

    return "Answer not found in document."

# ---------------- UI ----------------
uploaded_file = st.file_uploader("📄 Upload Study PDF", type=["pdf"])

if uploaded_file:

    text = extract_text(uploaded_file)

    if "Error reading PDF" in text:
        st.error(text)
    else:
        text = clean_text(text)

        st.subheader("📊 Document Overview")
        st.write("Total Characters:", len(text))

        # ---------------- SUMMARY ----------------
        st.subheader("🧠 Smart Summary")
        summary = generate_summary(text)
        st.info(summary)

        # ---------------- MCQ ----------------
        st.subheader("🎯 Practice MCQs")
        mcqs = generate_mcq(text)

        for i, q in enumerate(mcqs):
            st.write(f"{i+1}. {q}")

        # ---------------- FLASHCARDS ----------------
        st.subheader("🧾 Flashcards")
        cards = flashcards(text)

        for c in cards:
            with st.expander(c["front"]):
                st.write(c["back"])

        # ---------------- Q&A ----------------
        st.subheader("💬 Ask Doubt from Notes")

        user_q = st.text_input("Ask a question")

        if user_q:
            ans = answer_question(text, user_q)
            st.success(ans)

        # ---------------- DOWNLOAD REPORT ----------------
        report = f"""
AI STUDY REPORT
----------------

SUMMARY:
{summary}

MCQs:
{mcqs}

FLASHCARDS:
{cards}
"""

        st.download_button(
            "📥 Download Study Notes",
            report,
            file_name="study_report.txt"
        )
    
