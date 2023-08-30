import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

def make_table_data(res_list):
    labels = []
    seq = []
    scores = []
    for item in res_list:
        labels.append(item['labels'])
        seq.append(item['sequence'])
        scores.append(item['scores'])

    return seq, labels, scores

# Streamlit app
st.title('Text Classification')

# Get candidate labels from user
candidate_labels = st.text_input('Enter Candidate Labels (comma separated)', 'positive,negative,neutral')
candidate_labels = candidate_labels.split(",")

# File upload
uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df = df[['text']].dropna()

    if not candidate_labels:
        candidate_labels = ["positive", "negative", "neutral"]

    classifier = pipeline(
        task="zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    res = classifier(df['text'].tolist(), candidate_labels=candidate_labels)
    seq, labels, scores = make_table_data(res)

    classified_text_neg_pos = pd.DataFrame(list(zip(seq, labels, scores)), columns=['Text', 'Label', 'Score'])

    # Display a sample of the result
    st.write("Sample Results:")
    st.write(classified_text_neg_pos.head(3))

    # Display normalized results as a pie chart
    classified_text_normalized_neg_pos = pd.DataFrame(classified_text_neg_pos['Label'].value_counts(normalize=True))
    fig, ax = plt.subplots()
    ax.pie(classified_text_normalized_neg_pos['Label'], labels=classified_text_normalized_neg_pos.index, autopct='%1.1f%%')
    ax.axis('equal')
    st.write("Normalized Results:")
    st.pyplot(fig)
