import streamlit as st
import pandas as pd
from transformers import pipeline

# Function to process the text and classify it
def classify_text(file, candidate_labels):
    df = pd.read_excel(file)
    df = df[['text']].dropna()

    if not candidate_labels:
        candidate_labels = ["positive", "negative", "neutral"]

    classifier = pipeline(
        task="zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    res = classifier(df['text'].tolist(), candidate_labels=candidate_labels)

    labels = []
    seq = []
    scores = []

    for item in res:
        labels.append(item['labels'][0])  # Assuming the first label is the most likely one
        seq.append(item['sequence'])
        scores.append(item['scores'][0])  # Assuming the first score is the most likely one

    classified_text = pd.DataFrame(list(zip(seq, labels, scores)), columns=['Text', 'Label', 'Score'])
    classified_text_normalized = pd.DataFrame(classified_text['Label'].value_counts(normalize=True))

    return classified_text, classified_text_normalized

# Streamlit App
st.title('Text Classification')

# Upload the Excel file
file = st.file_uploader("Choose an Excel file containing 'text' column", type="xlsx")

# Input for candidate labels
candidate_labels_input = st.text_input("Candidate Labels (comma separated)", "positive,negative,neutral")
candidate_labels = [x.strip() for x in candidate_labels_input.split(",")]

if file:
    classified_text, classified_text_normalized = classify_text(file, candidate_labels)

    st.write("### Classified Text Sample")
    st.table(classified_text.head(3))

    st.write("### Label Distribution")
    st.bar_chart(classified_text_normalized)
