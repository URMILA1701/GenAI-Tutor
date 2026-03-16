import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
from src.rag_pipeline import ScienceTeacherRAG

@st.cache_resource
def load_rag():
    return ScienceTeacherRAG()

rag = load_rag()

st.title("📚 AI Science Teaching Assistant")
st.write("Prepare lesson explanations using the NCERT Class 10 Science textbook.")

query = st.text_area("Ask a teaching question:", height=100)

if st.button("Generate Teaching Plan"):

    if query.strip() != "":
        with st.spinner("Preparing lesson..."):
            answer,docs = rag.answer_query(query)

        st.subheader("Teaching Plan")
        st.write(answer)

        st.subheader("Retrieved Knowledge Base Context")
        for i, doc in enumerate(docs):
            st.write(f"Context {i+1}:")
            st.write(doc.page_content[:400])

    else:
        st.warning("Please enter a question.")