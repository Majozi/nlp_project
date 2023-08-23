import os
import streamlit as st
import pandas as pd
from transformers import pipeline

# Initialize summarization models
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
t5_summarizer = pipeline("summarization", model="t5-base")

def summarize_with_bart(text, min_length, max_length):
    summary = bart_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    return summary

def summarize_with_t5(text, min_length, max_length):
    summary = t5_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    return summary

# Sidebar options
st.sidebar.title("Summarization Parameters")
min_length = st.sidebar.slider("Minimum Length", 10, 100, 30)
max_length = st.sidebar.slider("Maximum Length", 50, 300, 100)

# Main app
st.title("Text Summarization Using BART and T5")
uploaded_file = st.file_uploader("Choose an Excel file containing 'text' column", type="xlsx")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)  # Replace with pd.read_excel for reading an Excel file
    df = df[['text']].dropna().astype('str')
    text_list = df['text'].tolist()
    
    bart_summaries = [summarize_with_bart(text, min_length, max_length) for text in text_list]
    t5_summaries = [summarize_with_t5(text, min_length, max_length) for text in text_list]
    
    result_df = pd.DataFrame({
        "Original Text": text_list,
        "BART Summary": bart_summaries,
        "T5 Summary": t5_summaries
    })
    
    st.write(result_df)
