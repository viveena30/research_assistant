# # summarizer.py

# import re
# import fitz  # PyMuPDF
# from nltk.tokenize import sent_tokenize
# import nltk
# import torch
# from transformers import (
#     AutoTokenizer, AutoModelForSeq2SeqLM,
#     BartTokenizer, BartForConditionalGeneration,
#     PegasusTokenizer, PegasusForConditionalGeneration
# )

# nltk.download("punkt")

# # ========== CONFIG ========== #
# pdf_path = "../pdfs/An efficient energy-aware approach for dynamic VM consolidation - 2021.pdf"     # Update this with your PDF path
# model_name = "Pegasus"     # Options: "DistilBART", "BART-Large-CNN", "Pegasus"
# # ============================= #

# # --------- Text Extraction --------- #
# def extract_text_pymupdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     return "\n".join([page.get_text("text") for page in doc])

# # --------- Preprocessing --------- #
# def deduplicate_sentences(text):
#     seen, unique = set(), []
#     sentences = re.split(r"(?<=[.?!])\s+", text)
#     for sentence in sentences:
#         clean = sentence.strip()
#         if clean and clean not in seen:
#             seen.add(clean)
#             unique.append(clean)
#     return " ".join(unique)

# def remove_references(text):
#     match = re.search(r"(?i)^references\s*$", text, re.MULTILINE)
#     return text[:match.start()].strip() if match else text

# def split_into_chunks(text, tokenizer, max_len=1024):
#     sentences = sent_tokenize(text)
#     chunks, chunk, length = [], "", 0

#     for sentence in sentences:
#         token_len = len(tokenizer.tokenize(sentence))
#         if length + token_len <= max_len:
#             chunk += sentence + " "
#             length += token_len
#         else:
#             chunks.append(chunk.strip())
#             chunk = sentence + " "
#             length = token_len
#     if chunk:
#         chunks.append(chunk.strip())
#     return chunks

# # --------- Summarization Models --------- #
# def summarize_model(text, tokenizer, model):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
#     chunks = split_into_chunks(text, tokenizer)

#     summary = ""
#     for chunk in chunks:
#         inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding="longest").to(device)
#         outputs = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
#         summary += tokenizer.decode(outputs[0], skip_special_tokens=True) + " "
#     return summary.strip()

# def summarize_distilbart(text):
#     model_id = "sshleifer/distilbart-cnn-12-6"
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
#     return summarize_model(text, tokenizer, model)

# def summarize_bart(text):
#     model_id = "facebook/bart-large-cnn"
#     tokenizer = BartTokenizer.from_pretrained(model_id)
#     model = BartForConditionalGeneration.from_pretrained(model_id)
#     return summarize_model(text, tokenizer, model)

# def summarize_pegasus(text):
#     model_id = "google/pegasus-xsum"
#     tokenizer = PegasusTokenizer.from_pretrained(model_id)
#     model = PegasusForConditionalGeneration.from_pretrained(model_id)
#     return summarize_model(text, tokenizer, model)

# def summarize_text_by_model(text, model_name):
#     if model_name == "DistilBART":
#         return summarize_distilbart(text)
#     elif model_name == "BART-Large-CNN":
#         return summarize_bart(text)
#     elif model_name == "Pegasus":
#         return summarize_pegasus(text)
#     else:
#         return "Invalid model selected."

# # --------- Section-wise Summarization --------- #
# def process_pdf_and_summarize(pdf_path, model_name):
#     raw_text = extract_text_pymupdf(pdf_path)
#     clean_text = deduplicate_sentences(remove_references(raw_text))

#     # Detect likely section headers
#     section_pattern = r"(?<=\n)(\d{0,2}[.)]?\s?(?:(?:[A-Z][a-z]+ ?)+|[A-Z][A-Z ]{3,}))(?=\n)"

#     # section_pattern = r"(?:^|\n)([A-Z][A-Z0-9 ,.\-]{3,}|[0-9]{1,2}[. ]+[A-Z][^\n]{3,})\n"
#     matches = list(re.finditer(section_pattern, clean_text))

#     if not matches:
#         print("No section headers detected. Summarizing full text...")
#         summary = summarize_text_by_model(clean_text, model_name)
#         print("\n### Summary ###\n", summary)
#         return

#     summaries = {}
#     for i, match in enumerate(matches):
#         section_title = match.group(1).strip()
#         section_start = match.end()
#         section_end = matches[i + 1].start() if i + 1 < len(matches) else len(clean_text)
#         section_text = clean_text[section_start:section_end].strip()

#         if section_text:
#             print(f"\nSummarizing section: {section_title}")
#             summary = summarize_text_by_model(section_text, model_name)
#             summaries[section_title] = summary

#     # Print section-wise summaries
#     print("\n--- Section-wise Summary ---\n")
#     for title, summary in summaries.items():
#         print(f"\n### {title} ###\n{summary}\n")

# # --------- Main Trigger --------- #
# if __name__ == "__main__":
#     process_pdf_and_summarize(pdf_path, model_name)





import fitz  # PyMuPDF
import re
from collections import OrderedDict

PDF_PATH = "../pdfs/An efficient energy-aware approach for dynamic VM consolidation - 2021.pdf"

STANDARD_SECTIONS = [
    "abstract", "introduction", "related work", "background", "literature review",
    "methodology", "materials and methods", "proposed method", "experiments",
    "results", "discussion", "conclusion", "future work", "acknowledgments", "references"
]

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def compile_standard_patterns():
    patterns = {}
    for sec in STANDARD_SECTIONS:
        pat = rf"(?:^|\n)\s*(\d+(\.\d+)*\s*)?{re.escape(sec)}[\s\n]*"
        patterns[sec.lower()] = re.compile(pat, re.IGNORECASE)
    return patterns

def detect_custom_headings(text):
    """
    Detect potential custom section titles based on heading heuristics.
    - Title case
    - Uppercase
    - Possibly numbered
    - Surrounded by newlines
    """
    candidates = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        clean = line.strip()

        # Heading-like line: title-case or all-caps and not too long
        if 5 < len(clean) < 100 and (
            clean.istitle() or clean.isupper()
        ):
            # Check it's likely a standalone heading
            prev = lines[i - 1].strip() if i > 0 else ""
            next = lines[i + 1].strip() if i < len(lines) - 1 else ""
            if prev == "" and next != "" and not clean.endswith("."):
                candidates.append((text.find(line), clean))
    return candidates

def find_all_section_positions(text):
    # Step 1: Find standard sections
    standard_patterns = compile_standard_patterns()
    matches = []

    for name, pattern in standard_patterns.items():
        for m in pattern.finditer(text):
            matches.append((m.start(), name))

    # Step 2: Find custom headings
    custom = detect_custom_headings(text)
    for pos, title in custom:
        matches.append((pos, title.strip()))

    # Deduplicate by position, sort
    seen = set()
    unique_matches = []
    for pos, name in sorted(matches, key=lambda x: x[0]):
        if pos not in seen:
            unique_matches.append((pos, name))
            seen.add(pos)
    return unique_matches

def extract_sections(text, matches):
    sections = OrderedDict()
    for i, (start, name) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections[name] = content
    return sections

def pretty_print_sections(sections):
    for title, content in sections.items():
        print(f"\n{'='*20} {title.upper()} {'='*20}")
        print(content[:1000].strip())  # Truncate for readability
        print("\n... [truncated]\n")

if __name__ == "__main__":
    print("[+] Reading and parsing PDF...")
    full_text = extract_text_from_pdf(PDF_PATH)

    print("[+] Finding all section headers (standard + custom)...")
    section_positions = find_all_section_positions(full_text)

    if not section_positions:
        print("[-] No valid sections found.")
    else:
        print(f"[+] Found {len(section_positions)} sections. Extracting content...\n")
        structured = extract_sections(full_text, section_positions)
        pretty_print_sections(structured)
