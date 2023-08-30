import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
from transformers import pipeline
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Downloading the NLTK resources if not downloaded
nltk.download('stopwords')
nltk.download('wordnet')



def thematic_analysis(file):
    df = pd.read_excel(file)  # Reading the Excel file
    df = df[['text']].dropna()

    stoplist = stopwords.words('english')
    c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2, 4))
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

if selection == 'Getting Started':
    st.title("Natural Language Processing")
    st.write("""

Natural Language Processing (NLP) is a multifaceted field that integrates computer science, artificial intelligence, and linguistics to facilitate the interaction between computers and human language. In the context of higher education research, NLP plays a vital role in analyzing and synthesizing vast amounts of textual data. Researchers leverage NLP techniques to automatically grade assignments, detect plagiarism, and extract meaningful insights from academic texts. Moreover, NLP supports the summarization of extensive literature, the assessment of language proficiency, and the personalization of learning experiences. These applications not only enhance the efficiency and effectiveness of educational practices but also open new avenues for exploration and innovation in higher education. Through its ability to understand and process natural language, NLP is revolutionizing the way higher education institutions conduct research, teach, and engage with students.
    """)

elif selection == 'Sentiment':
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
Group the items in the table below to get the themes. To flesh them out in your discussion, go back to your original data and search these words to get more insight. When creating a theme, 
remember to get a sum of all the bigrams/trigrams that you combined so that you may Quantify your argument. \n \n **PLEASE NOTE THIS**: The table below doesn't represent the number of responses, but the number of times the
bigrams/trigrams occur on your data.
""")
    uploaded_file = st.file_uploader("Choose an Excel file containing 'text' column", type="xlsx")
    
    if uploaded_file is not None:
        df_ngram = thematic_analysis(uploaded_file)
        st.write(df_ngram)

