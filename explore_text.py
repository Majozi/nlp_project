import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import networkx as nx
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Function to check if a text is meaningless
def is_meaningless(text):
    return bool(re.fullmatch(r'[0-9\s\W]*', text))

# Function to extract the first sentence for summary
def extractive_summarize(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences[0] if sentences else "No text to summarize"

# Function to plot the graph
def plot_graph(filtered_feedback, cosine_sim, threshold):
    fig, ax = plt.subplots(figsize=(12, 12))
    G = nx.Graph()
    
    for i in range(len(filtered_feedback)):
        G.add_node(i, text=filtered_feedback[i])

    for i in range(len(filtered_feedback)):
        for j in range(i+1, len(filtered_feedback)):
            if cosine_sim[i, j] > threshold:
                G.add_edge(i, j, weight=cosine_sim[i, j])

    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_color='skyblue', node_size=500, font_size=8, font_color='black', edge_color='gray', ax=ax)
    ax.set_title('Network Graph of Feedback Based on Cosine Similarity')
    st.pyplot(fig)

# Initialize vectorizers
count_vectorizer = CountVectorizer(stop_words='english')
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Streamlit App
st.title('Text Analysis Dashboard')

# File Upload
uploaded_file = st.file_uploader("Upload your file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    if 'text' not in df.columns:
        st.error("The uploaded file does not contain a column named 'text'. Please upload a valid file.")
    else:
        feedback_text = df['text'].dropna().astype(str).tolist()
        
        # 6. Network Graph Analysis
        st.subheader('Network Graph Analysis')
        threshold = st.slider('Set Similarity Threshold', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        filtered_feedback = [text for text in df['text'] if isinstance(text, str) and 'tutor' in text.lower()]
        tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_feedback)
        cosine_sim = cosine_similarity(tfidf_matrix)
        plot_graph(filtered_feedback, cosine_sim, threshold)
