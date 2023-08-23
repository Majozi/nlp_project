import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import re

# Downloading the NLTK resources if not downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Topic Modelling Page
st.title('Topic Modelling')

# Upload file
uploaded_file = st.file_uploader("Choose an Excel file containing 'text' column", type="xlsx")

# Input for number of topics
number_of_topics = st.slider("Select the Number of Topics", min_value=1, max_value=10, value=3)

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = df[['text']].dropna()

    my_stopwords = stopwords.words('english')
    lemmatizer = WordNetLemmatizer().lemmatize
    my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'

    # cleaning master function
    def clean_text(text, bigrams=False):
        text = text.lower() # lower case
        text = re.sub('['+my_punctuation + ']+', ' ', text) # strip punctuation
        text = re.sub('\s+', ' ', text) # remove double spacing
        text = re.sub('([0-9]+)', '', text) # remove numbers
        text_token_list = [word for word in text.split(' ') if word not in my_stopwords] # remove stopwords
        text_token_list = [lemmatizer(word) if '#' not in word else word for word in text_token_list] # apply word rooter
        if bigrams:
            text_token_list = text_token_list + [text_token_list[i]+'_'+text_token_list[i+1] for i in range(len(text_token_list)-1)]
        text = ' '.join(text_token_list)
        return text

    df['clean_feeds'] = df.text.apply(clean_text)

    # the vectorizer object will be used to transform text to vector form
    vectorizer = CountVectorizer(max_df=0.9, min_df=5, token_pattern='\w+|\$[\d\.]+|\S+')
    tf = vectorizer.fit_transform(df['clean_feeds']).toarray()
    tf_feature_names = vectorizer.get_feature_names_out()

    model = LatentDirichletAllocation(n_components=number_of_topics, random_state=0)
    model.fit(tf)

    def display_topics(model, feature_names, no_top_words):
        topic_dict = {}
        for topic_idx, topic in enumerate(model.components_):
            topic_dict["Topic %d words" % (topic_idx)] = ['{}'.format(feature_names[i]) for i in topic.argsort()[:-no_top_words - 1:-1]]
            topic_dict["Topic %d weights" % (topic_idx)] = ['{:.1f}'.format(topic[i]) for i in topic.argsort()[:-no_top_words - 1:-1]]
        return pd.DataFrame(topic_dict)

    no_top_words = 20
    topics = display_topics(model, tf_feature_names, no_top_words)

    # Displaying the topics
    st.write(topics)
