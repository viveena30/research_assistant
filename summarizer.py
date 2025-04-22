# summarizer.py

import re
import fitz  # PyMuPDF
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from collections import Counter

import nltk
nltk.download("punkt")

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    BartTokenizer, BartForConditionalGeneration,
    PegasusTokenizer, PegasusForConditionalGeneration
)
import torch

# Extract text using PyMuPDF
def extract_text_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc])

# Deduplicate sentences
def deduplicate_sentences(text):
    seen, unique = set(), []
    sentences = re.split(r"(?<=[.?!])\s+", text)
    for sentence in sentences:
        clean = sentence.strip()
        if clean and clean not in seen:
            seen.add(clean)
            unique.append(clean)
    return " ".join(unique)

# Remove "References" and after
def remove_references(text):
    match = re.search(r"(?i)^references\s*$", text, re.MULTILINE)
    return text[:match.start()].strip() if match else text

# Preprocess into chunks
def split_into_chunks(text, tokenizer, max_len=1024):
    sentences = sent_tokenize(text)
    chunks, chunk, length = [], "", 0

    for sentence in sentences:
        token_len = len(tokenizer.tokenize(sentence))
        if length + token_len <= max_len:
            chunk += sentence + " "
            length += token_len
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "
            length = token_len
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# DistilBART
def summarize_distilbart(text):
    model_id = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return summarize_model(text, tokenizer, model)

# BART-Large-CNN
def summarize_bart(text):
    model_id = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_id)
    model = BartForConditionalGeneration.from_pretrained(model_id)
    return summarize_model(text, tokenizer, model)

# Pegasus
def summarize_pegasus(text):
    model_id = "google/pegasus-xsum"
    tokenizer = PegasusTokenizer.from_pretrained(model_id)
    model = PegasusForConditionalGeneration.from_pretrained(model_id)
    return summarize_model(text, tokenizer, model)

# Generic summarizer function
def summarize_model(text, tokenizer, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    chunks = split_into_chunks(text, tokenizer)

    summary = ""
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding="longest").to(device)
        outputs = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
        summary += tokenizer.decode(outputs[0], skip_special_tokens=True) + " "
    return summary.strip()

# Full cleaning + summarization
def process_pdf_and_summarize(pdf_path, model_name):
    raw_text = extract_text_pymupdf(pdf_path)
    clean_text = deduplicate_sentences(remove_references(raw_text))

    if model_name == "DistilBART":
        return summarize_distilbart(clean_text)
    elif model_name == "BART-Large-CNN":
        return summarize_bart(clean_text)
    elif model_name == "Pegasus":
        return summarize_pegasus(clean_text)
    else:
        return "Invalid model selected."
