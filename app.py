import streamlit as st
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Learning Companion", layout="wide")
st.title("📚 AI Learning Companion")

text_input = st.text_area("Paste your study material here")

if text_input:
    text = re.sub(r'\s+', ' ', text_input).strip()

    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) == 0:
        st.error("No usable text found.")
        st.stop()

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences)

    st.subheader("Summary")
    st.write(" ".join(sentences[:5]))
