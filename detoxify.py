import streamlit as st
import pandas as pd
from detoxify import Detoxify

# Title and description
st.title("Detoxify Streamlit App")
st.write("Upload a CSV/Excel file with a column named 'text' to get toxicity predictions.")

# Upload file
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read file into DataFrame
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        
        # Check if 'text' column exists
        if 'text' not in df.columns:
            st.error("No 'text' column found in the uploaded file.")
        else:
            # Make predictions using Detoxify
            texts = df['text'].tolist()
            results = Detoxify('original').predict(texts)
            
            # Add results to the DataFrame
            for key in results.keys():
                df[key] = results[key]

            # Display the updated DataFrame
            st.write(df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
