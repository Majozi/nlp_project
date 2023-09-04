import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
from transformers import pipeline
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import altair as alt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Downloading the NLTK resources if not downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Function to make DataFrame downloadable
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="classified_text.csv">Download CSV File</a>'

# Function to make Pie Chart downloadable
def get_pie_chart_download_link(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="pie_chart.png">Download Pie Chart</a>'

# Function to download N-GRAMS data as a CSV file
def download_csv(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="ngrams.csv">Download CSV File</a>'
    return href

def thematic_analysis(file, ngram_min, ngram_max):
    df = pd.read_excel(file)
    df = df[['text']].dropna()
    
    stoplist = stopwords.words('english')
    c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(ngram_min, ngram_max))
    ngrams = c_vec.fit_transform(df['text'])
    count_values = ngrams.toarray().sum(axis=0)
    vocab = c_vec.vocabulary_

    df_ngram = pd.DataFrame(sorted([(count_values[i], k) for k, i in vocab.items()], reverse=True)).rename(columns={0: 'frequency', 1: 'ngram'})

    return df_ngram

# Image URL
image_url = "https://www.up.ac.za/themes/up2.0/images/vertical-logo-bg.png"

# Displaying the image
st.image(image_url, width=100)

# Top Navigation
st.sidebar.title('Text Analytics')
selection = st.sidebar.radio("Go to", ['Getting Started', 'Summarization', 'Sentiment', 'Toxicity', 'N-Grams (Thematic)', 'Text Classification', 'Topic Modelling'])

if selection == 'Getting Started':
    st.title("Natural Language Processing")
    st.write("""

Natural Language Processing (NLP) is a multifaceted field that integrates computer science, artificial intelligence, and linguistics to facilitate the interaction between computers and human language. In the context of higher education research, NLP plays a vital role in analyzing and synthesizing vast amounts of textual data. Researchers leverage NLP techniques to automatically grade assignments, detect plagiarism, and extract meaningful insights from academic texts. Moreover, NLP supports the summarization of extensive literature, the assessment of language proficiency, and the personalization of learning experiences. These applications not only enhance the efficiency and effectiveness of educational practices but also open new avenues for exploration and innovation in higher education. Through its ability to understand and process natural language, NLP is revolutionizing the way higher education institutions conduct research, teach, and engage with students.
    """)

elif selection == 'Sentiment':
    st.title("Sentiment Analysis")

    # Text Classification pipeline
    classifier = pipeline('sentiment-analysis')
    
    # Streamlit App
    st.title('Text Classification App')
    st.write("""
             **TIPS FOR USE:** \n After the analysis has been completed, download the table below and read through a sample of responses to get a feel of
             how accurate the classification was. A rule of thumb is to always pay attention to rows where the 
             score is below 50 and the sentiment is negative and the score is close to 100%. \n \n
             An overall positive sentiment that is below 50% (depending on the question asked), may be
             an immediate indicator of issues to be addressed. To find these, filter for a score above 75%
             and a negative label. \n \n
             Be mindful that you may have to reclassify and then recalculate the overall sentiment.
             """)
    
    # File Upload
    uploaded_file = st.file_uploader("Upload CSV or Excel with a column name 'text'.", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            df = pd.read_excel(uploaded_file)
    
        if 'text' not in df.columns:
            st.markdown('File does not have a `text` column. Please upload another.')
        else:
            # Perform Text Classification
            df['label'] = df['text'].apply(lambda x: classifier(x)[0]['label'])
            df['score'] = df['text'].apply(lambda x: classifier(x)[0]['score']) * 100
    
            st.write(df)
    
            # Download DataFrame
            st.markdown(get_table_download_link(df), unsafe_allow_html=True)
    
            # Sentiment Percentage
            sentiment_counts = df['label'].value_counts(normalize=True)
    
            # Define colors
            colors = ['#005baa' if label == 'POSITIVE' else '#c48939' for label in sentiment_counts.index]
    
            # Pie Chart
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
            st.pyplot(fig)
    
            # Download Pie Chart
            st.markdown(get_pie_chart_download_link(fig), unsafe_allow_html=True)
    
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
elif selection == 'Topic Modelling':
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
            text = text.lower()  # lower case
            text = re.sub('[' + re.escape(my_punctuation) + ']+', ' ', text)  # strip punctuation
            text = re.sub('\s+', ' ', text)  # remove double spacing
            text = re.sub('([0-9]+)', '', text)  # remove numbers
            text_token_list = [word for word in text.split(' ')
                               if word not in my_stopwords]  # remove stopwords

            text_token_list = [word_rooter(word) if '#' not in word else word
                               for word in text_token_list]  # apply word rooter
            if bigrams:
                text_token_list = text_token_list + [text_token_list[i] + '_' + text_token_list[i + 1]
                                                     for i in range(len(text_token_list) - 1)]
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
                topic_dict["Topic %d words" % (topic_idx)] = ['{}'.format(feature_names[i])
                                                              for i in topic.argsort()[:-no_top_words - 1:-1]]
                topic_dict["Topic %d weights" % (topic_idx)] = ['{:.1f}'.format(topic[i])
                                                                for i in topic.argsort()[:-no_top_words - 1:-1]]
            return pd.DataFrame(topic_dict)

        topics = display_topics(model, tf_feature_names, no_top_words)
        st.write(topics)

elif selection == 'N-Grams (Thematic)':
    st.title("Thematic Analysis Using N-Grams")
    st.write("""
    **How to use this analysis:** \n Once the analysis is done, group similar ideas in the table below to get the themes. You can also copy the table and paste it to ChatGPT and use a prompt to get the themes. To flesh them out in your discussion, go back to your original data and search these words to get more insight. When creating a theme, 
    remember to get a sum of all the bigrams/trigrams that you combined so that you may Quantify your argument. \n \n **PLEASE NOTE THIS**: The table below doesn't represent the number of responses, but the number of times the
    bigrams/trigrams occur on your data.
    """)
    uploaded_file = st.file_uploader("Choose an Excel file containing 'text' column", type="xlsx")
    
    ngram_min = st.slider("Minimum N-gram Range", 1, 5, 2)
    ngram_max = st.slider("Maximum N-gram Range", ngram_min, 5, 4)
    
    if uploaded_file is not None:
        df_ngram = thematic_analysis(uploaded_file, ngram_min, ngram_max)
        st.write(df_ngram)
    
        if df_ngram is not None and not df_ngram.empty:
            top_ngrams = df_ngram.head(25)
    
            chart = alt.Chart(top_ngrams).mark_bar().encode(
                y=alt.Y('ngram:O', sort='-x'),
                x='frequency:Q',
                tooltip=['ngram', 'frequency']
            ).properties(
                title='Top 25 N-grams',
                width=600
            )
    
            st.altair_chart(chart)  
