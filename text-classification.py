import streamlit as st
import pandas as pd
from transformers import pipeline

# Function to process the text and classify it
def classify_text(file, candidate_labels):
    try:
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
            labels.append(item['labels'][0])
            seq.append(item['sequence'])
            scores.append(item['scores'][0])

        classified_text = pd.DataFrame(list(zip(seq, labels, scores)), columns=['Text', 'Label', 'Score'])
        classified_text_normalized = pd.DataFrame(classified_text['Label'].value_counts(normalize=True))

        return classified_text, classified_text_normalized
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return None, None

# Streamlit App
st.title('Text Classification')

# Upload the Excel file
file = st.file_uploader("Choose an Excel file containing 'text' column", type="xlsx")

# Input for candidate labels
candidate_labels_input = st.text_input("Candidate Labels (comma separated)", "positive,negative,neutral")
candidate_labels = [x.strip() for x in candidate_labels_input.split(",")]

if file:
    classified_text, classified_text_normalized = classify_text(file, candidate_labels)

    if classified_text is not None:
        st.write("### Classified Text Sample")
        st.table(classified_text.head(3))

    if classified_text_normalized is not None:
        st.write("### Label Distribution")
        st.bar_chart(classified_text_normalized)
