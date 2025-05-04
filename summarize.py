import os
import json
import requests
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import clean
from unstructured.documents.elements import Title

HF_TOKEN = os.environ.get("HF_TOKEN")  # Set this in GitHub secrets
PDF_PATH = os.environ.get("PDF_PATH", "input.pdf")
MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # You can change this if needed

# Extract elements from PDF
elements = partition_pdf(PDF_PATH)
sections = []
current_section = {"title": "Introduction", "content": ""}

for el in elements:
    if isinstance(el, Title):
        if current_section["content"].strip():
            sections.append(current_section)
        current_section = {"title": el.text.strip(), "content": ""}
    else:
        current_section["content"] += el.text + "\n"

if current_section["content"].strip():
    sections.append(current_section)

# Summarization function using Hugging Face Inference API
def summarize_with_llama(text, section_title):
    API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    prompt = f"Summarize the following section titled '{section_title}':\n\n{text}"
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300}}
    
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        return response.json()[0]["generated_text"]
    except Exception as e:
        print("Error:", e)
        print("Response:", response.text)
        return f"Could not summarize section: {section_title}"

# Generate the summary
output_lines = []
for section in sections:
    print(f"Summarizing section: {section['title']}")
    summary = summarize_with_llama(section["content"], section["title"])
    output_lines.append(f"## {section['title']}\n\n{summary}\n\n")

# Save to file
with open("output_summary.txt", "w", encoding="utf-8") as f:
    f.writelines(output_lines)

print("✅ Summary saved to output_summary.txt")