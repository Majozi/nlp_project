import streamlit as st
import pandas as pd
from transformers import pipeline

# Title and description
st.title("Zero-Shot Classification Streamlit App")
st.write("Upload a CSV or Excel file with a column named 'text' to get zero-shot classification results.")

# Initialize zero-shot-classification pipeline
classifier = pipeline("zero-shot-classification", model="distilbert-base-uncased-finetuned-mnli")

# User-defined labels for classification
label1 = st.text_input("Enter Label 1:")
label2 = st.text_input("Enter Label 2:")
label3 = st.text_input("Enter Label 3:")
label4 = st.text_input("Enter Label 4:")
label5 = st.text_input("Enter Label 5:")

labels = [label for label in [label1, label2, label3, label4, label5] if label]

# Upload file
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read the uploaded file into a DataFrame
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        
        # Check if 'text' column exists
        if 'text' not in df.columns:
            st.error("No 'text' column found in the uploaded file.")
        else:
            results_list = []
            
            # Perform zero-shot classification and store the results
            for text in df['text']:
                result = classifier(text, labels)
                results_list.append(result['labels'][0])  # Most likely label
                
            # Add the results to the DataFrame
            df['classification_result'] = results_list
            
            # Display the updated DataFrame
            st.write(df)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
