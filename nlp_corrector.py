#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import transformers
from transformers import pipeline
import nltk
import re

# Download NLTK data for tokenization
nltk.download('punkt')

# Load the Hugging Face models
grammar_corrector = pipeline("text2text-generation", model="facebook/bart-large-cnn")
paraphraser = pipeline("text2text-generation", model="t5-base")

# Function to correct grammar and punctuation
def correct_grammar(text):
    response = grammar_corrector(text, max_length=512, num_return_sequences=1)
    return response[0]['generated_text']

# Function to paraphrase text
def paraphrase_text(text):
    sentences = nltk.sent_tokenize(text)  # Break into sentences for better results
    paraphrased_sentences = []
    for sentence in sentences:
        # Process each sentence using the paraphrasing pipeline
        response = paraphraser(f"paraphrase: {sentence} </s>", max_length=512, num_return_sequences=1)
        paraphrased_sentences.append(response[0]['generated_text'])
    return " ".join(paraphrased_sentences)

# Function to clean text
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# Streamlit UI
st.title("AI_Powered Ammar Jamshed Grammar Correction and Plagiarism-Proofing App")
st.write("Upload your text file or paste your content to get plagiarism-proof, grammatically correct writing.")

# File upload or text input
uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
input_text = ""

if uploaded_file:
    # Read uploaded file
    input_text = uploaded_file.read().decode("utf-8")
elif st.text_area("Paste your text here:"):
    input_text = st.text_area("Paste your text here:")

if input_text:
    # Clean input text
    input_text = clean_text(input_text)

    # Correct grammar
    st.subheader("Step 1: Grammar Correction")
    corrected_text = correct_grammar(input_text)
    st.write(corrected_text)

    # Paraphrase text
    st.subheader("Step 2: Plagiarism-Proof Paraphrasing")
    paraphrased_text = paraphrase_text(corrected_text)
    st.write(paraphrased_text)

    # Option to download the result
    st.download_button(
        "Download Corrected & Paraphrased Text",
        paraphrased_text,
        file_name="corrected_paraphrased_text.txt",
    )


# In[ ]:




