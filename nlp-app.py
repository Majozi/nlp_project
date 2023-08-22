import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# Summarization Function
def summarize_text(text):
    bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    t5_summarizer = pipeline("summarization", model="t5-base")
    # You can customize the summarization logic here
    summarized_text = bart_summarizer(text)[0]['summary_text']
    return summarized_text

# Topic Modeling Function
def perform_topic_modeling(text):
    # Preprocess and vectorize the text
    vectorizer = CountVectorizer()
    text_vectorized = vectorizer.fit_transform([text])
    # Perform topic modeling using LDA
    lda_model = LatentDirichletAllocation(n_components=5)
    lda_model.fit(text_vectorized)
    topics = vectorizer.get_feature_names_out()
    return topics

# Sentiment Analysis Function
def perform_sentiment_analysis(text):
    model_name = "siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # Analyze sentiment (customize as needed)
    sentiment_results = "Positive" # Example result
    return sentiment_results

# Title
st.title('NLP Analysis App')

# File uploader
uploaded_file = st.file_uploader("Choose a text file", type=['txt', 'csv', 'xlsx'])

# Operation selector
operation = st.selectbox('Choose an operation:', ['Summarization', 'Topic Modeling', 'Sentiment Analysis'])

# Trigger buttons
if st.button('Start Analysis'):
    if uploaded_file is not None:
        # Read the uploaded file
        if uploaded_file.type == 'text/plain':
            uploaded_text = uploaded_file.read().decode()
        elif uploaded_file.type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'text/csv']:
            df = pd.read_excel(uploaded_file) if uploaded_file.type.endswith('sheet') else pd.read_csv(uploaded_file)
            uploaded_text = ' '.join(df['text'].dropna().astype('str')) # Assuming the text is in the 'text' column

        st.write(f"Performing {operation}...")
        if operation == 'Summarization':
            summarized_text = summarize_text(uploaded_text)
            st.write("Summarized Text:", summarized_text)
        elif operation == 'Topic Modeling':
            topics = perform_topic_modeling(uploaded_text)
            st.write("Topics:", topics)
        elif operation == 'Sentiment Analysis':
            sentiment_results = perform_sentiment_analysis(uploaded_text)
            st.write("Sentiment Analysis Results:", sentiment_results)
    else:
        st.write("Please upload a file first.")
