import pandas as pd
import streamlit as st
from transformers import pipeline

# Zero-Shot Classification pipeline
classifier = pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli')

# Streamlit App
st.title('Text Classification App')

# Define user input for Classification Labels
labels = st.text_input('Enter your classification labels, separated by comma')

# Split the labels into a list
labels = [label.strip() for label in labels.split(',')]

# File Upload
uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        df = pd.read_excel(uploaded_file)

    if 'text' not in df.columns:
        st.markdown('File does not have a `text` column. Please upload another.')
    else:
        def classify(text, labels):
            result = classifier(text, labels)
            return pd.Series([result['labels'][0], result['scores'][0]], index=['label', 'score'])
        
        res_df = df['text'].apply(lambda x: classify(x, labels))
        df = pd.concat([df, res_df], axis=1)
        
        st.write(df)
