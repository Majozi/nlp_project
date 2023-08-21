import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, Trainer, BertTokenizer, BertModel
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import io

# Initialize summarization models
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
t5_summarizer = pipeline("summarization", model="t5-base")

# Define functions
def summarize_with_bart(text, min_length, max_length):
    summary = bart_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    return summary

def summarize_with_t5(text, min_length, max_length):
    summary = t5_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    return summary

# Page title and sidebar
st.title("NLP Analysis")
page = st.sidebar.selectbox("Choose an analysis", ["Home", "Summarize", "Thematic", "Classification", "Topic Modelling", "Sentiment", "FAQs", "Toxicity"])

# Home page
if page == "Home":
    st.write("Welcome to the NLP Analysis tool!")

# Summarize page
elif page == "Summarize":
    st.write("Summarize")
    uploaded_file = st.file_uploader("Choose an Excel file")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df = df[['text']].dropna()
        text = df['text'].tolist()
        min_length = st.number_input("Minimum Length", value=50)
        max_length = st.number_input("Maximum Length", value=200)
        bart_summary = summarize_with_bart(text, min_length, max_length)
        t5_summary = summarize_with_t5(text, min_length, max_length)
        st.write("BART Summary:", bart_summary)
        st.write("T5 Summary:", t5_summary)

# Thematic page
elif page == "Thematic":
    st.write("Thematic Analysis")
    # Add your thematic analysis code here

# Classification page
elif page == "Classification":
    st.write("Classification")
    # Add your classification code here

# Topic Modelling page
elif page == "Topic Modelling":
    st.write("Topic Modelling")
    # Add your topic modelling code here

# Sentiment page
elif page == "Sentiment":
    st.write("Sentiment Analysis")
    # Add your sentiment analysis code here

# FAQs page
elif page == "FAQs":
    st.write("Frequently Asked Questions")
    # Add your FAQs content here

# Toxicity page
elif page == "Toxicity":
    st.write("Toxicity Analysis")
    # Add your toxicity analysis code here
