import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
import os
from transformers import pipeline

# Downloading the NLTK stopwords list if not downloaded
nltk.download('stopwords')

# Function for thematic analysis
def thematic_analysis(file, ngram_range):
    df = pd.read_excel(file)
    df = df[['text']].dropna()

    stoplist = stopwords.words('english')
    c_vec = CountVectorizer(stop_words=stoplist, ngram_range=ngram_range)
    ngrams = c_vec.fit_transform(df['text'])
    count_values = ngrams.toarray().sum(axis=0)
    vocab = c_vec.vocabulary_

    df_ngram = pd.DataFrame(sorted([(count_values[i], k) for k, i in vocab.items()],
                                   reverse=True)).rename(columns={0: 'frequency', 1: 'bigram/trigram'})

    return df_ngram

# Image URL
image_url = "https://www.up.ac.za/themes/up2.0/images/vertical-logo-bg.png"

# Displaying the image
st.image(image_url, width=100)

# Top Navigation
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", ['Getting Started', 'Summarization', 'Sentiment', 'Toxicity', 'N-Grams (Thematic)', 'Text Classification', 'Topic Modelling'])

# Getting Started Page
if selection == 'Getting Started':
    st.title('Getting Started with Large Language Models')
    st.write("""
    Large language models, particularly transformers, have revolutionized natural language processing (NLP). 
    They enable various applications like translation, summarization, sentiment analysis, and more. 
    Transformers are deep learning models that process words in relation to all other words in a sentence,
    capturing complex relationships and structures. They're pre-trained on vast datasets and can be fine-tuned
    for specific tasks. This introduction provides an overview of their capabilities and uses.
    """)

# Summarization Page
elif selection == 'Summarization':
    st.title('Text Summarization')
    st.write("""
    Text summarization is the process of condensing a larger piece of text into a concise summary. 
    It helps in extracting the essential information from a document, preserving only the most 
    critical points. Summarization techniques can be abstractive or extractive, with large language 
    models playing a significant role in generating human-like summaries.
    """)

# Sentiment Page
elif selection == 'Sentiment':
    st.title('Sentiment Analysis')
    st.write("""
    Sentiment analysis refers to the use of natural language processing to identify and categorize the sentiment
    expressed in a piece of text. It can detect whether the sentiment is positive, negative, or neutral, and is widely
    used in social media monitoring, customer feedback, and market research.
    """)

# Toxicity Page
elif selection == 'Toxicity':
    st.title('Toxicity Detection')
    st.write("""
    Toxicity detection is essential in moderating online discussions. It involves identifying 
    and filtering out toxic or harmful content, such as hate speech, abusive language, or misinformation. 
    Machine learning models, including transformers, have become vital tools in automating this process.
    """)

# N-Grams (Thematic) Page
elif selection == 'N-Grams (Thematic)':
    st.title('N-Grams (Thematic) Analysis')
    st.write("""
    N-Grams are continuous sequences of n items from a given text or speech. Thematic analysis using N-Grams 
    helps in understanding the context, themes, and frequently occurring patterns in a text. It's a useful 
    technique in text mining and natural language processing.
    """)

    ngram_min = st.sidebar.slider("Minimum N-Gram", 1, 5, 3)
    ngram_max = st.sidebar.slider("Maximum N-Gram", ngram_min, 5, 4)
    ngram_range = (ngram_min, ngram_max)

    uploaded_file = st.file_uploader("Choose an Excel file containing 'text' column", type="xlsx")
    if uploaded_file is not None:
        df_ngram = thematic_analysis(uploaded_file, ngram_range)
        st.write(df_ngram)

# Text Classification Page
elif selection == 'Text Classification':
    st.title('Text Classification')

    candidate_labels_input = st.text_input("Enter candidate labels, separated by commas (e.g., positive,negative,neutral):")
    candidate_labels = candidate_labels_input.split(",") if candidate_labels_input else []

    uploaded_file = st.file_uploader("Choose an Excel file containing 'text' column", type="xlsx")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df = df[['text']].dropna()

        if not candidate_labels:
            # If no candidate labels are provided, use default labels
            candidate_labels = ["positive", "negative", "neutral"]

        classifier = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")

        res = classifier(df['text'].tolist(), candidate_labels=candidate_labels)

        def make_table_data(res_list):
            labels = []
            seq = []
            scores = []
            for item in res_list:
                labels.append(item['labels'][0])
                seq.append(item['sequence'])
                scores.append(item['scores'][0])

            return seq, labels, scores

        seq, labels, scores = make_table_data(res)

        classified_text_neg_pos = pd.DataFrame(list(zip(seq, labels, scores)), columns=['Text', 'Label', 'Score'])
        classified_text_normalized_neg_pos = pd.DataFrame(classified_text_neg_pos['Label'].value_counts(normalize=True))

        st.write(classified_text_neg_pos)
        st.write(classified_text_normalized_neg_pos)

# Topic Modelling Page
st.title('Topic Modelling')

# Upload file
uploaded_file = st.file_uploader("Choose an Excel file containing 'text' column", type="xlsx")

# Input for min and max value of topics
min_topics = st.slider("Select the Minimum Number of Topics", min_value=1, max_value=10, value=1)
max_topics = st.slider("Select the Maximum Number of Topics", min_value=min_topics, max_value=20, value=5)
no_top_words = st.slider("Select the Number of Top Words", min_value=1, max_value=50, value=20)

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = df[['text']].dropna()

    my_stopwords = stopwords.words('english')
    word_rooter = WordNetLemmatizer().lemmatize
    my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

    # cleaning master function
    def clean_text(text, bigrams=False):
        text = text.lower() # lower case
        text = re.sub('['+my_punctuation + ']+', ' ', text) # strip punctuation
        text = re.sub('\s+', ' ', text) #remove double spacing
        text = re.sub('([0-9]+)', '', text) # remove numbers
        text_token_list = [word for word in text.split(' ')
                                if word not in my_stopwords] # remove stopwords

        text_token_list = [word_rooter(word) if '#' not in word else word
                            for word in text_token_list] # apply word rooter
        if bigrams:
            text_token_list = text_token_list+[text_token_list[i]+'_'+text_token_list[i+1]
                                                for i in range(len(text_token_list)-1)]
        text = ' '.join(text_token_list)
        return text

    df['clean_feeds'] = df.text.apply(clean_text)

    # the vectorizer object will be used to transform text to vector form
    vectorizer = CountVectorizer(token_pattern='\w+|\$[\d\.]+|\S+')
    # apply transformation
    tf = vectorizer.fit_transform(df['clean_feeds']).toarray()
    # tf_feature_names tells us what word each column in the matrix represents
    tf_feature_names = vectorizer.get_feature_names_out()

    model = LatentDirichletAllocation(n_components=max_topics, random_state=0)
    model.fit(tf)

    def display_topics(model, feature_names, no_top_words):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                            for i in topic.argsort()[:-no_top_words - 1:-1]]
            topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                            for i in topic.argsort()[:-no_top_words - 1:-1]]
        return pd.DataFrame(topic_dict)

    topics = display_topics(model, tf_feature_names, no_top_words)

    st.write(topics)
