import streamlit as st
from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

st.title('Sentiment Analysis')
user_input = st.text_area("Enter your text here:")

if st.button('Analyze'):
    sentiment_score = analyze_sentiment(user_input)
    if sentiment_score > 0:
        st.write('Positive Sentiment')
    elif sentiment_score < 0:
        st.write('Negative Sentiment')
    else:
        st.write('Neutral Sentiment')
