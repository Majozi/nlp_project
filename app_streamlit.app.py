import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pandas as pd

# Title
st.title('Text Analysis')

# Sidebar selection
selection = st.sidebar.selectbox("Choose an Analysis Type", ['Thematic Analysis', 'Summarization'])

if selection == 'Thematic Analysis':
    st.title('Thematic Analysis')
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        df = df[['text']].dropna()

        stoplist = stopwords.words('english')
        c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(3, 4))
        ngrams = c_vec.fit_transform(df['text'])
        count_values = ngrams.toarray().sum(axis=0)
        vocab = c_vec.vocabulary_

        df_ngram = pd.DataFrame(sorted([(count_values[i], k) for k, i in vocab.items()],
                                       reverse=True)).rename(columns={0: 'frequency', 1: 'bigram/trigram'})

        # Save as an Excel file
        downloads_path = st.text_input('Enter the download path:')
        if downloads_path:
            with pd.ExcelWriter(downloads_path + r'\nlp_analysis.xlsx') as writer:
                df_ngram.to_excel(writer, sheet_name='thematic')
            st.success("File saved successfully!")

elif selection == 'Summarization':
    st.title('Text Summarization')
    st.write("""
    Text summarization is the process of condensing a larger piece of text into a concise summary. 
    It helps in extracting the essential information from a document, preserving only the most 
    critical points. Summarization techniques can be abstractive or extractive, with large language 
    models playing a significant role in generating human-like summaries.
    """)
