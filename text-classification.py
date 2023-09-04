import pandas as pd
import streamlit as st
from transformers import pipeline

# Initialize Zero-Shot Classification pipeline
classifier = pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli')

# Streamlit App Title
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
        if df.empty:
            st.markdown('Uploaded file is empty. Please upload another.')
            raise ValueError("Empty File")
    except Exception as e:
        try:
            df = pd.read_excel(uploaded_file)
            if df.empty:
                st.markdown('Uploaded file is empty. Please upload another.')
                raise ValueError("Empty File")
        except Exception as e:
            st.markdown(f'Error reading file: {e}')
            raise

    # Drop empty rows based on the 'text' column
    df.dropna(subset=['text'], inplace=True)

    if 'text' not in df.columns:
        st.markdown('File does not have a `text` column. Please upload another.')
    elif not labels:
        st.markdown('Please enter some classification labels.')
    else:
        def classify(text, labels):
            result = classifier(text, labels)
            return pd.Series([result['labels'][0], result['scores'][0]], index=['label', 'score'])
        
        df[['label', 'score']] = df['text'].apply(lambda x: classify(x, labels))
        st.write(df)
