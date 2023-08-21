import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import pandas as pd

# Top Navigation
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", ['Getting Started', 'Summarization', 'Sentiment', 'Toxicity', 'N-Grams (Thematic)', 'Text Classification', 'Topic Modelling'])

# Getting Started Page
if selection == 'Getting Started':
    st.title('Getting Started with Large Language Models')
    st.write("""
    Large language models, particularly transformers, have revolutionized natural language processing (NLP). 
    They enable various applications like translation, summarization, sentiment analysis, and more. 
    Transformers are deep learning models that process words in relation to all other words in a sentence,
    capturing complex relationships and structures. They're pre-trained on vast datasets and can be fine-tuned
    for specific tasks. This introduction provides an overview of their capabilities and uses.
    """)

# Summarization Page
elif selection == 'Summarization':
    st.title('Text Summarization')
    st.write("""
    Text summarization is the process of condensing a larger piece of text into a concise summary. 
    It helps in extracting the essential information from a document, preserving only the most 
    critical points. Summarization techniques can be abstractive or extractive, with large language 
    models playing a significant role in generating human-like summaries.
    """)

# Sentiment Page
elif selection == 'Sentiment':
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

# Toxicity Page
elif selection == 'Toxicity':
    st.title('Toxicity Detection')
    st.write("""
    Toxicity detection is essential in moderating online discussions. It involves identifying 
    and filtering out toxic or harmful content, such as hate speech, abusive language, or misinformation. 
    Machine learning models, including transformers, have become vital tools in automating this process.
    """)

# N-Grams (Thematic) Page
elif selection == 'N-Grams (Thematic)':
    st.title('N-Grams (Thematic) Analysis')
    st.write("""
    N-Grams are continuous sequences of n items from a given text or speech. Thematic analysis using N-Grams 
    helps in understanding the context, themes, and frequently occurring patterns in a text. It's a useful 
    technique in text mining and natural language processing.
    """)

# Text Classification Page
elif selection == 'Text Classification':
    st.title('Text Classification')
    st.write("""
    Text classification is the task of categorizing text into predefined classes or labels. 
    It includes applications like spam detection, topic labeling, sentiment analysis, and more. 
    Machine learning models, especially large language models, have excelled in this domain.
    """)

# Topic Modelling Page
elif selection == 'Topic Modelling':
    st.title('Topic Modelling')
    st.write("""
    Topic modeling is a technique to discover the hidden thematic structure in a large collection of documents.
    It's incredibly useful in organizing, understanding and summarizing large datasets of textual information.
    Algorithms like LDA (Latent Dirichlet Allocation) are commonly used for this purpose.
    """)
