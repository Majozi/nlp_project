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
def is_meaningless(text):
    return bool(re.fullmatch(r'[0-9\s\W]*', text))

def extractive_summarize(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences[0] if sentences else "No text to summarize"

def plot_graph(filtered_feedback, cosine_sim, threshold):
    G = nx.Graph()
    for i, feedback in enumerate(filtered_feedback):
        G.add_node(i, text=feedback)
    for i in range(len(filtered_feedback)):
        for j in range(i+1, len(filtered_feedback)):
            if cosine_sim[i, j] > threshold:
                G.add_edge(i, j, weight=cosine_sim[i, j])
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'text'), node_color='skyblue', node_size=500, font_size=8, font_color='black', edge_color='gray')
    plt.title('Network Graph of Student Feedback About Tutor Based on Cosine Similarity')
    st.pyplot()
st.title('Text Analysis Dashboard')
uploaded_file = st.file_uploader("Upload your file (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if 'text' not in df.columns:
        st.error("The uploaded file does not contain a column named 'text'. Please upload a valid file.")
    else:
        feedback_text = df['text'].dropna().astype(str).tolist()

        # User Input for additional stopwords
        additional_stopwords = st.text_input("Enter additional stopwords separated by commas:")
        if additional_stopwords:
            additional_stopwords = additional_stopwords.split(',')
            additional_stopwords = [word.strip().lower() for word in additional_stopwords]

        count_vectorizer = CountVectorizer(stop_words='english')
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        
        if additional_stopwords:
            new_stopwords = set(count_vectorizer.get_stop_words()).union(set(additional_stopwords))
            count_vectorizer.set_params(stop_words=new_stopwords)
            tfidf_vectorizer.set_params(stop_words=new_stopwords)
        st.subheader('1. Word Cloud Analysis')
        wc_width = st.slider("Word Cloud Width", 400, 1200, 800)
        wc_height = st.slider("Word Cloud Height", 200, 800, 400)
        
        filtered_words = ' '.join(feedback_text).split()
        filtered_words = [word for word in filtered_words if word.lower() not in set(count_vectorizer.get_stop_words())]
        word_freq = Counter(filtered_words)
        wordcloud = WordCloud(width=wc_width, height=wc_height, background_color='white').generate_from_frequencies(word_freq)
        
        st.image(wordcloud.to_array(), caption='Word Cloud of Feedback', use_column_width=True)
        st.subheader('2. Binned Word Count Analysis')
        bin_labels = st.multiselect("Select Word Count Bins", ['0-5', '6-10', '11-20', '21-50', '51-100', '101+'], default=['0-5', '6-10', '11-20', '21-50', '51-100', '101+'])
        
        df['word_count'] = df['text'].dropna().apply(lambda x: len(x.split()))
        bins = [0, 5, 10, 20, 50, 100, np.inf]
        labels = ['0-5', '6-10', '11-20', '21-50', '51-100', '101+']
        
        df['word_count_bin'] = pd.cut(df['word_count'], bins=bins, labels=labels, right=False)
        word_count_bins = df[df['word_count_bin'].isin(bin_labels)]['word_count_bin'].value_counts().sort_index()
        
        fig1, ax1 = plt.subplots()
        sns.barplot(x=word_count_bins.index, y=word_count_bins.values, palette='viridis', ax=ax1)
        ax1.set_title("Distribution of Word Counts in Feedback")
        ax1.set_xlabel("Word Count Range")
        ax1.set_ylabel("Number of Responses")
        for p in ax1.patches:
            ax1.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline')
        st.pyplot(fig1)
        st.subheader('3. Meaningless Responses Check')
        st.write('This section identifies responses that are meaningless (e.g., consist only of numbers, whitespace, or special characters).')
        df['is_meaningless'] = df['text'].apply(lambda x: is_meaningless(str(x)) if pd.notna(x) else False)
        meaningless_count = df['is_meaningless'].sum()
        st.write(f"Number of meaningless responses: {meaningless_count}")

        st.subheader('4. Extractive Summary')
        st.write('This section provides an extractive summary by selecting the first sentence from the concatenated feedback text.')
        all_texts = " ".join(df['text'].astype(str))
        extractive_summary = extractive_summarize(all_texts)
        st.write("Extractive Summary:")
        st.write(extractive_summary)

        st.subheader('5. Top Most Important Words Using TF-IDF')
        top_n_words = st.slider('Select Top N Words', 5, 50, 20)
        
        tfidf_matrix = tfidf_vectorizer.fit_transform(feedback_text)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=0)
       
