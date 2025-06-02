import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

st.set_page_config(page_title="ProcureBot PH", layout="wide")
st.title("📜 ProcureBot PH")
st.caption("Legal Chatbot for RA 12009 and its IRR")

# Sidebar for file source
source_option = st.sidebar.radio("📂 Choose PDF Source", ["Use sample PDFs", "Upload your own"])

# Upload or load sample PDFs
if source_option == "Use sample PDFs":
    file_paths = [
        "sample_pdfs/RA_12009_searchable.pdf",
        "sample_pdfs/IRR_RA_12009_searchable.pdf"
    ]
    pdfs = [fitz.open(path) for path in file_paths]
else:
    uploaded_files = st.file_uploader("📄 Upload searchable PDFs", type="pdf", accept_multiple_files=True)
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
        st.stop()
    pdfs = [fitz.open(stream=pdf.read(), filetype="pdf") for pdf in uploaded_files]

query = st.text_input("🔍 Ask your procurement law question:")

@st.cache_data(show_spinner=False)
def extract_paragraphs(pdfs):
    all_paragraphs = []
    for doc in pdfs:
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n\n"
        paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 50]
        all_paragraphs.extend(paragraphs)
    return all_paragraphs

if query:
    st.info("⚙️ Extracting text and searching...")
    paragraphs = extract_paragraphs(pdfs)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(paragraphs, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    query_embedding = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, k=3)

    st.subheader("📘 Relevant Legal Sections")
    for i in indices[0]:
        st.markdown(f"> {paragraphs[i]}")
        st.markdown("---")
