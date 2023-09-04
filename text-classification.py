import pandas as pd
import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Initialize Zero-Shot Classification pipeline
classifier = pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli')

# Streamlit App Title
st.title('Text Classification App')

# Custom CSS for download buttons
st.markdown("""
<style>
.download-btn {
    background-color: #c48939;
    color: white;
    padding: 14px 20px;
    margin: 8px 0;
    border: none;
    cursor: pointer;
    width: 100%;
    text-align: center;
}
.download-btn:hover {
    background-color: #005baa;
}
</style>
""", unsafe_allow_html=True)

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

        # Normalize the labels and calculate frequency distribution
        label_counts = df['label'].value_counts(normalize=True)
        
        # Sort by frequency
        label_counts = label_counts.sort_values(ascending=False)
        
        # Pie chart
        fig, ax = plt.subplots()
        ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=['#005baa', '#c48939', '#d61c33'])
        ax.axis('equal')
        
        # Remove background
        ax.set_facecolor('none')
        fig.patch.set_visible(False)

        # Convert the Matplotlib figure to a BytesIO object
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Convert DataFrame to CSV and encode
        csv = df.to_csv(index=False)
        csv_base64 = base64.b64encode(csv.encode()).decode()

        # Custom download buttons
        st.markdown(f'<a href="data:text/csv;base64,{csv_base64}" download="classified_data.csv" class="download-btn"><i class="fas fa-download"></i> Download Data as CSV</a>', unsafe_allow_html=True)
        st.markdown(f'<a href="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}" download="label_distribution.png" class="download-btn"><i class="fas fa-download"></i> Download Pie Chart</a>', unsafe_allow_html=True)

        # Display pie chart
        st.pyplot(fig)

        # Display DataFrame
        st.write(df)
