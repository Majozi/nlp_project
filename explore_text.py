
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


# Functions
def is_meaningless(text):
    return bool(re.fullmatch(r'[0-9\s\W]*', text))

# Function to plot the graph
def plot_graph(filtered_feedback, cosine_sim, threshold):
    G = nx.Graph()
    for i in range(len(filtered_feedback)):
        G.add_node(i, text=filtered_feedback[i])

    for i in range(len(filtered_feedback)):
        for j in range(i+1, len(filtered_feedback)):
            if cosine_sim[i, j] > threshold:
                G.add_edge(i, j, weight=cosine_sim[i, j])

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'text'), node_color='skyblue', node_size=500, font_size=8, font_color='black', edge_color='gray')
    plt.title('Network Graph of Feedback')
    st.pyplot()

# Streamlit App
st.title('Text Analysis Dashboard')

# File Upload
uploaded_file = st.file_uploader("Upload your file (CSV or Excel)", type=["csv", "xlsx"])
additional_stopwords = st.text_input("Enter additional stopwords separated by commas: ", '').split(',')

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if 'text' not in df.columns:
        st.error("The uploaded file does not contain a column named 'text'. Please upload a valid file.")
    else:
        # 1. Word Cloud Analysis
        st.subheader('1. Word Cloud Analysis')
        wc_width = st.slider("Word Cloud Width", 400, 1200, 800)
        wc_height = st.slider("Word Cloud Height", 200, 800, 400)
        
        feedback_text = df['text'].dropna().astype(str).tolist()
        filtered_words = ' '.join(feedback_text).split()
        filtered_words = [word for word in filtered_words if word.lower() not in set(CountVectorizer(stop_words='english').get_stop_words())]
        filtered_words = [word for word in filtered_words if word.lower() not in additional_stopwords]
        word_freq = Counter(filtered_words)
        wordcloud = WordCloud(width=wc_width, height=wc_height, background_color='white').generate_from_frequencies(word_freq)
        st.image(wordcloud.to_array(), caption='Word Cloud of Feedback', use_column_width=True)
        
        # 2. Binned Word Count Analysis
        st.subheader('2. Binned Word Count Analysis')
        bin_labels = st.multiselect("Select Word Count Bins", ['0-5', '6-10', '11-20', '21-50', '51-100', '101+'], default=['0-5', '6-10', '11-20', '21-50', '51-100', '101+'])
        df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
        bins = [0, 5, 10, 20, 50, 100, np.inf]
        labels = ['0-5', '6-10', '11-20', '21-50', '51-100', '101+']
        df['word_count_bin'] = pd.cut(df['word_count'], bins=bins, labels=labels, right=False)
        word_count_bins = df[df['word_count_bin'].isin(bin_labels)]['word_count_bin'].value_counts().sort_index()
        fig, ax1 = plt.subplots()
        sns.barplot(x=word_count_bins.index, y=word_count_bins.values, palette='viridis', ax=ax1)
        ax1.set_title("Distribution of Word Counts in Feedback")
        ax1.set_xlabel("Word Count Range")
        ax1.set_ylabel("Number of Responses")
        st.pyplot(fig)
        
        # 3. Meaningless Responses Check
        st.subheader('3. Meaningless Responses Check')
        df['is_meaningless'] = df['text'].apply(lambda x: is_meaningless(str(x)) if pd.notna(x) else False)
        meaningless_count = df['is_meaningless'].sum()
        st.write(f"Number of meaningless responses: {meaningless_count}")

        # 4. Important Words Using TF-IDF
        st.subheader('4. Top Most Important Words Using TF-IDF')
        top_n_words = st.slider('Select Top N Words', 5, 50, 20)
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(feedback_text)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=0)
        sorted_tfidf = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)[:top_n_words]
        words = [word[0] for word in sorted_tfidf]
        tfidf_values = [word[1] for word in sorted_tfidf]
        fig2, ax2 = plt.subplots()
        bars = ax2.barh(words, tfidf_values, color='purple')
        ax2.set_xlabel('TF-IDF Score')
        ax2.set_ylabel('Words')
        ax2.set_title(f'Top {top_n_words} Words by TF-IDF Score')
        ax2.invert_yaxis()
        st.pyplot(fig2)

        # 5. Network Graph Analysis
        st.subheader('5. Network Graph Analysis')
        threshold = st.slider('Set Similarity Threshold', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        filtered_feedback = [text for text in feedback_text if 'tutor' in text.lower()]
        tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_feedback)
        cosine_sim = cosine_similarity(tfidf_matrix)
        plot_graph(filtered_feedback, cosine_sim, threshold)

