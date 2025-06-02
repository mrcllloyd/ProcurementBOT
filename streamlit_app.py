import streamlit as st
from chatbot_engine import semantic_search

st.set_page_config("ProcureBot PH", layout="wide")
st.title("📜 ProcureBot PH (TXT-based)")
st.caption("Ask questions from RA 12009 and its IRR – now powered by semantic AI search")

query = st.text_input("🔍 Ask your procurement law question:")

if query:
    with st.spinner("Searching legal provisions..."):
        results = semantic_search(query)
    st.subheader("📘 Top Matches")
    for i, (source, text) in enumerate(results, 1):
        st.markdown(f"**{i}. Source: `{source}`**")
        st.markdown(f"> {text}")
        st.markdown("---")
