import streamlit as st
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# Make sure to download the NLTK stopwords list if you haven't already
import nltk
nltk.download('stopwords')

def thematic_analysis(file, ngram_min, ngram_max):
    df = pd.read_excel(file)  # Reading the Excel file
    df = df[['text']].dropna()

    stoplist = stopwords.words('english')
    c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(ngram_min, ngram_max))
    ngrams = c_vec.fit_transform(df['text'])
    count_values = ngrams.toarray().sum(axis=0)
    vocab = c_vec.vocabulary_

    df_ngram = pd.DataFrame(sorted([(count_values[i], k) for k, i in vocab.items()],
                                   reverse=True)).rename(columns={0: 'frequency', 1: 'ngram'})

    return df_ngram

# Main app
st.title("Thematic Analysis Using N-Grams")
uploaded_file = st.file_uploader("Choose an Excel file containing 'text' column", type="xlsx")

# Sliders for n-gram range
ngram_min = st.slider("Minimum N-gram Range", 1, 5, 2)
ngram_max = st.slider("Maximum N-gram Range", ngram_min, 5, 4)  # Make sure max is always >= min

st.write("""
Group the items in the table below to get the themes. To flesh them out in your discussion, go back to your original data and search these words to get more insight. When creating a theme, 
remember to get a sum of all the bigrams/trigrams that you combined so that you may quantify your argument. PLEASE NOTE THIS: The table below doesn't represent the number of responses, but the number of times the
n-grams occur in your data. \n \n Use the slider to control the number of 'grams'.
""")

if uploaded_file is not None:
    df_ngram = thematic_analysis(uploaded_file, ngram_min, ngram_max)
    st.write(df_ngram)
    # Plotting the top 15 n-grams
    top_ngrams = df_ngram.head(15)
    plt.figure(figsize=(10, 6))
    plt.barh(top_ngrams['ngram'], top_ngrams['frequency'], color='skyblue')
    plt.xlabel('Frequency')
    plt.ylabel('N-grams')
    plt.title('Top 15 N-grams')
    plt.gca().invert_yaxis()
    st.pyplot()
