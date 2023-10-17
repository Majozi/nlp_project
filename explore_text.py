import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import pandas as pd

# Functions
def is_meaningless(text):
    if pd.isna(text) or not isinstance(text, str):
        return False
    return bool(re.fullmatch(r'[0-9\s\W]*', text))

# Additional Functions
def create_word_cloud(counter, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(counter)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    st.pyplot()

# Initialize
count_vectorizer = CountVectorizer(stop_words='english')
stop_words = set(count_vectorizer.get_stop_words())
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Streamlit App
st.title('Text Analysis Dashboard')

# File Upload
uploaded_file = st.file_uploader("Upload your file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if 'text' not in df.columns:
        st.error("The uploaded file does not contain a column named 'text'. Please upload a valid file.")
    else:
        # User Inputs
        additional_stopwords = st.text_input("Enter additional stopwords separated by commas: ").split(',')
        additional_stopwords = [word.strip().lower() for word in additional_stopwords]
        stop_words.update(additional_stopwords)
        
        drop_rows = st.selectbox("Do you want to drop rows containing only numbers or symbols?", ("Yes", "No"))
        if drop_rows == 'Yes':
            num_or_sym_rows = df['text'].apply(is_meaningless)
            df.drop(df[num_or_sym_rows].index, inplace=True)

        # Data Preprocessing
        feedback_text = df['text'].dropna().astype(str).tolist()
        
        # Initialize Counters
        word_counter = Counter()
        stemmed_word_counter = Counter()
        lemmatized_word_counter = Counter()

        # Populate the Counters
        for text in feedback_text:
            words = text.split()
            word_counter.update(words)
            stemmed_words = [stemmer.stem(word) for word in words]
            stemmed_word_counter.update(stemmed_words)
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            lemmatized_word_counter.update(lemmatized_words)

        # Filter stopwords
        filtered_stemmed_word_counter = {word: count for word, count in stemmed_word_counter.items() if word.lower() not in stop_words}
        filtered_lemmatized_word_counter = {word: count for word, count in lemmatized_word_counter.items() if word.lower() not in stop_words}
        
        # Generate Word Clouds
        st.subheader('Word Clouds')
        create_word_cloud(filtered_stemmed_word_counter, 'Stemmed Data')
        create_word_cloud(filtered_lemmatized_word_counter, 'Lemmatized Data')

        # Continue with the rest of your original code (e.g., Word Cloud Analysis, Binned Word Count Analysis, etc.)
