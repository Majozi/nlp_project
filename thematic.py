import streamlit as st
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# Make sure to download the NLTK stopwords list if you haven't already
import nltk
nltk.download('stopwords')

def thematic_analysis(file):
    df = pd.read_excel(file)  # Reading the Excel file
    df = df[['text']].dropna()

    stoplist = stopwords.words('english')
    c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2, 4))
    ngrams = c_vec.fit_transform(df['text'])
    count_values = ngrams.toarray().sum(axis=0)
    vocab = c_vec.vocabulary_

    df_ngram = pd.DataFrame(sorted([(count_values[i], k) for k, i in vocab.items()],
                                   reverse=True)).rename(columns={0: 'frequency', 1: 'bigram/trigram'})

    return df_ngram

# Main app
st.title("Thematic Analysis Using N-Grams")
uploaded_file = st.file_uploader("Choose an Excel file containing 'text' column", type="xlsx")

if uploaded_file is not None:
    df_ngram = thematic_analysis(uploaded_file)
    st.write(df_ngram)
