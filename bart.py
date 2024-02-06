# Import necessary libraries
import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from PyPDF2 import PdfReader

# Load the BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Function to extract text from a PDF file
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to summarize text using BART
def summarize_text(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length= 1024, truncation=True)
    summary_ids = model.generate(inputs, max_length = 1024,min_length=50, length_penalty=1.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app
st.title("PDF Summarizer with BART")

# File upload functionality
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Extract text from the uploaded PDF file
    text_from_pdf = extract_text_from_pdf(uploaded_file)

    # Summarize the extracted text
    summary = summarize_text(text_from_pdf)

    # Display the summarized text
    # st.subheader("Original Text:")
    # st.text(text_from_pdf)
    st.subheader("Summarized Text:")
    st.text(summary)



