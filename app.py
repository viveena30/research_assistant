# app.py

import streamlit as st
import os
from summarizer import process_pdf_and_summarize

# Directory to store uploads
UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(page_title="PDF Summarizer", layout="centered")

st.title("📄 PDF Summarizer with Transformers")
st.markdown("Upload a PDF file and choose a model to generate a concise summary.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
model_choice = st.selectbox("Select Summarization Model", ["DistilBART", "BART-Large-CNN", "Pegasus"])

if uploaded_file and model_choice:
    save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {uploaded_file.name}")

    if st.button("Summarize"):
        with st.spinner("Generating summary... Please wait ⏳"):
            summary = process_pdf_and_summarize(save_path, model_choice)
            st.subheader("📝 Summary")
            st.write(summary)
