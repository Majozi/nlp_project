import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# Text Classification
def classify_text(text):
    classifier = pipeline("text-classification", model="bert-base-uncased")
    classification = classifier(text)
    return classification

# Toxicity Analysis
def analyze_toxicity(text):
    toxicity_analyzer = pipeline("text-classification", model="unitary/toxic-bert") # Replace with a toxicity model
    toxicity_analysis = toxicity_analyzer(text)
    return toxicity_analysis

# Summarization
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summarized_text = summarizer(text)[0]['summary_text']
    return summarized_text

# Topic Modeling
def perform_topic_modeling(text):
    vectorizer = CountVectorizer()
    text_vectorized = vectorizer.fit_transform([text])
    lda_model = LatentDirichletAllocation(n_components=5)
    lda_model.fit(text_vectorized)
    topics = vectorizer.get_feature_names_out()
    return topics

# Thematic Analysis (n-grams)
def perform_thematic_analysis(text):
    vectorizer = CountVectorizer(ngram_range=(2,2)) # 2-grams
    ngrams = vectorizer.fit_transform([text])
    ngrams_list = vectorizer.get_feature_names_out()
    return ngrams_list

# Sentiment Analysis
def perform_sentiment_analysis(text):
    model_name = "siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    sentiment_results = sentiment_analyzer(text)
    return sentiment_results

# Rest of the Streamlit code remains the same


# Title
st.title('NLP Analysis App')

# File uploader
uploaded_file = st.file_uploader("Choose a text file", type=['txt', 'csv', 'xlsx'])

# Operation selector
operation = st.selectbox('Choose an operation:', ['Text Classification', 'Toxicity Analysis', 'Summarization', 'Topic Modeling', 'Thematic Analysis (n-grams)', 'Sentiment Analysis'])

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
        if operation == 'Text Classification':
            classified_text = classify_text(uploaded_text)
            st.write("Classified Text:", classified_text)
        elif operation == 'Toxicity Analysis':
            toxicity_analysis = analyze_toxicity(uploaded_text)
            st.write("Toxicity Analysis:", toxicity_analysis)
        elif operation == 'Summarization':
            summarized_text = summarize_text(uploaded_text)
            st.write("Summarized Text:", summarized_text)
        elif operation == 'Topic Modeling':
            topics = perform_topic_modeling(uploaded_text)
            st.write("Topics:", topics)
        elif operation == 'Thematic Analysis (n-grams)':
            thematic_analysis = perform_thematic_analysis(uploaded_text)
            st.write("Thematic Analysis:", thematic_analysis)
        elif operation == 'Sentiment Analysis':
            sentiment_results = perform_sentiment_analysis(uploaded_text)
            st.write("Sentiment Analysis Results:", sentiment_results)
    else:
        st.write("Please upload a file first.")
