import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer
import numpy as np

st.title("Sentiment Analysis")

uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    local_csv_dataset = df[['text']]
    pred_texts = local_csv_dataset.dropna().astype('str')
    pred_texts = pred_texts['text'].tolist()

    model_name = "siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    trainer = Trainer(model=model)

    tokenized_texts = tokenizer(pred_texts, truncation=True, padding=True)

    class SimpleDataset:
        def __init__(self, tokenized_texts):
            self.tokenized_texts = tokenized_texts

        def __len__(self):
            return len(self.tokenized_texts["input_ids"])

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.tokenized_texts.items()}

    pred_dataset = SimpleDataset(tokenized_texts)
    predictions = trainer.predict(pred_dataset)
    preds = predictions.predictions.argmax(-1)
    labels = pd.Series(preds).map(model.config.id2label)
    scores = (np.exp(predictions[0]) / np.exp(predictions[0]).sum(-1, keepdims=True)).max(1)
    sentiment = pd.DataFrame(list(zip(pred_texts, preds, labels, scores)), columns=['text', 'Prediction', 'Label', 'Score'])

    # Display the sentiment analysis results
    st.write(sentiment)

    # Optionally, you can save the results to an Excel file
    if st.button("Download as Excel"):
        download_path = "nlp_analysis.xlsx"
        with pd.ExcelWriter(download_path) as writer:
            sentiment.to_excel(writer, sheet_name='sentiment')
        st.success(f"File saved to {download_path}")
