import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import pandas as pd

import streamlit as st

html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" rel="stylesheet">
</head>
<body style="margin-top: 200px;">
    <nav class="navbar navbar-expand-lg navbar-light" style="height: 100px; width: 100%; background-color: #005baa; position: fixed; top: 50px; left: 0; z-index: 1000;">
        <div class="container d-flex justify-content-between align-items-center" style="padding: 20px 10px 0 10px;">
            <div style="text-align: center; font-size: 25px; font-weight: bold; color: white;">Natural Language Processing</div>
            <img src="https://www.up.ac.za/themes/up2.0/images/vertical-logo-bg.png" alt="Logo" style="height: 90px; width: 90px" />
            <div style="text-align: center; font-size: 25px; font-weight: bold; color: white;">Exploring open-source models</div>
        </div><br><br><br>
    </nav>
</body>
</html>
"""

st.markdown(html_code, unsafe_allow_html=True)

# Title and Introduction about NLP
st.title('')
st.write("""
Natural Language Processing (NLP) is a field of artificial intelligence that enables computers to understand, interpret, and generate human language. This technology allows machines to interact with text data, perform various analyses, and extract meaningful insights. The following app provides different NLP techniques to analyze and process text data.
""")

# Text Classification
def classify_text(text, labels):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    classification = classifier(text, labels)
    return classification

# Summarization
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    return summarized_text
# Topic Modeling
def perform_topic_modeling(text):
    vectorizer = CountVectorizer()
    text_vectorized = vectorizer.fit_transform([text])
    best_topics = 3 # Customize the logic to find the best number of topics
    lda_model = LatentDirichletAllocation(n_components=best_topics)
    lda_model.fit(text_vectorized)
    topics = vectorizer.get_feature_names_out()
    return topics

# Thematic Analysis (n-grams)
def perform_thematic_analysis(text):
    vectorizer = CountVectorizer(ngram_range=(2,2))
    ngrams = vectorizer.fit_transform([text])
    ngrams_list = vectorizer.get_feature_names_out()
    ngrams_count = ngrams.sum(axis=0)
    return ngrams_list, ngrams_count

# File uploader
uploaded_file = st.file_uploader("Choose a text file", type=['txt', 'csv', 'xlsx'])

# Operation selector
operation = st.selectbox('Choose an operation:', ['Text Classification', 'Summarization', 'Topic Modeling', 'Thematic Analysis (n-grams)'])

# Trigger buttons
if st.button('Start Analysis'):
    if uploaded_file is not None:
        # Read the uploaded file based on its type
        if uploaded_file.type == 'text/plain':
            uploaded_text = uploaded_file.read().decode()
        elif uploaded_file.type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'text/csv']:
            df = pd.read_excel(uploaded_file) if uploaded_file.type.endswith('sheet') else pd.read_csv(uploaded_file)
            uploaded_text = ' '.join(df['text'].dropna().astype('str')) # Assuming the text is in the 'text' column
        else:
            st.warning("Unsupported file type")
            uploaded_text = ""
        
        st.write(f"Performing {operation}...")
        if operation == 'Text Classification':
            user_labels = st.text_input("Enter the labels for classification (comma-separated):")
            labels = user_labels.split(',')
            classified_text = classify_text(uploaded_text, labels)
            st.write("Classified Text:", classified_text)
        elif operation == 'Summarization':
            summarized_text, negative_count, positive_count = summarize_text(uploaded_text)
            st.write("Summarized Text:", summarized_text)
        elif operation == 'Topic Modeling':
            topics = perform_topic_modeling(uploaded_text)
            st.write("Topics:", topics)
        elif operation == 'Thematic Analysis (n-grams)':
            ngrams_list, ngrams_count = perform_thematic_analysis(uploaded_text)
            st.write("N-grams Identified:", ngrams_list)
            st.write("Frequency Count:", ngrams_count)
    else:
        st.write("Please upload a file first.")

# NLP Techniques Table
nlp_techniques = pd.DataFrame({
    'Analysis Type': ['Text Classification', 'Summarization', 'Topic Modeling', 'Thematic Analysis (n-grams)'],
    'Summary': ['Classify text into user-defined labels', 'Summarize long texts into concise summaries', 'Identify main topics in a text', 'Analyze frequent word combinations (n-grams)'],
    'Suggestions on how to use': ['Categorize documents, Sentiment analysis', 'Content summarization, Highlight generation', 'Content categorization, Theme discovery', 'Text pattern analysis, Keyword extraction']
})
st.table(nlp_techniques)
