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
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from transformers import DistilBertForSequenceClassification
import networkx as nx
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import openai
import types

# Custom hash function
def ignore_hash(*args, **kwargs):
    return 0

@st.cache(allow_output_mutation=True, hash_funcs={types.FunctionType: ignore_hash})
def load_model():
    return pipeline('zero-shot-classification', model='typeform/distilbert-base-uncased-mnli')

# Initialize Zero-Shot Classification pipeline
classifier = load_model()

# Downloading the NLTK resources if not downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Chunking Function
def chunk_text(text, chunk_size=800):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# OpenAI API functions with chunking
def analyze_text(api_key, text, prompt):
    openai.api_key = api_key
    chunks = chunk_text(text)
    complete_response = []
    for chunk in chunks:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt + chunk,
            max_tokens=150
        )
        complete_response.append(response.choices[0].text.strip())
    return ' '.join(complete_response)

def summarize_text(api_key, text):
    openai.api_key = api_key
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt="Summarize this text with 100.:\n\n" + chunk,
            max_tokens=150
        )
        summaries.append(response.choices[0].text.strip())
    return ' '.join(summaries)

def analyze_sentiment(api_key, text):
    openai.api_key = api_key
    chunks = chunk_text(text)
    sentiments = []
    for chunk in chunks:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt="Determine the sentiment of this text (positive, neutral, negative):\n\n" + chunk,
            max_tokens=60
        )
        sentiments.append(response.choices[0].text.strip().lower())
    # Aggregate sentiments (can be adjusted based on requirements)
    if "negative" in sentiments:
        return "Negative"
    elif "positive" in sentiments:
        return "Positive"
    else:
        return "Neutral"

def detect_toxicity(api_key, text):
    openai.api_key = api_key
    chunks = chunk_text(text)
    toxicities = []
    for chunk in chunks:
        response = openai.Completion.create(
            engine="content-filter-alpha-c4",
            prompt=chunk,
            max_tokens=1,
            temperature=0,
            top_p=0
        )
        toxicities.append(response.choices[0].text)
    if '2' in toxicities:
        return '2'
    else:
        return '0'

# ... [Rest of your existing code for Streamlit UI and logic]


# Initialize nltk
nltk.download('wordnet')
nltk.download('stopwords')

# Image URL
image_url = "https://www.up.ac.za/themes/up2.0/images/vertical-logo-bg.png"

# Displaying the image
st.image(image_url, width=100)

# Top Navigation
st.sidebar.title('Text Analytics')
selection = st.sidebar.radio("Go to", ['Getting Started', 'Text Exploration','Sentiment', 'N-Grams (Thematic)', 'Text Classification', 'Topic Modelling', 'Combined Analysis'])

if selection == 'Getting Started':
    st.title("Natural Language Processing")
    st.write("""

Natural Language Processing (NLP) is a multifaceted field that integrates computer science, artificial intelligence, and linguistics to facilitate the interaction between computers and human language. In the context of higher education research, NLP plays a vital role in analyzing and synthesizing vast amounts of textual data. Researchers leverage NLP techniques to automatically grade assignments, detect plagiarism, and extract meaningful insights from academic texts. Moreover, NLP supports the summarization of extensive literature, the assessment of language proficiency, and the personalization of learning experiences. These applications not only enhance the efficiency and effectiveness of educational practices but also open new avenues for exploration and innovation in higher education. Through its ability to understand and process natural language, NLP is revolutionizing the way higher education institutions conduct research, teach, and engage with students.
    """)

elif selection == 'Sentiment':
    st.title("Sentiment Analysis")

    
    classifier = pipeline('sentiment-analysis')
    
    st.write("""
         **TIPS FOR USE:** \n After the analysis has been completed, download the table below and read through a sample of responses to get a feel of
         how accurate the classification was. A rule of thumb is to always pay attention to rows where the 
         score is below 50 and the sentiment is negative and the score is close to 100%. \n 
         An overall positive sentiment that is below 50% (depending on the question asked), may be
         an immediate indicator of issues to be addressed. To find these, filter for a score above 75%
         and a negative label. \n
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
            # Drop NaN or empty values and ensure the 'text' column contains strings
            df.dropna(subset=['text'], inplace=True)
            df['text'] = df['text'].astype(str)
    
            # Perform Text Classification in a batch
            classified_text = classifier(df['text'].tolist())
            
            df['label'] = [item['label'] for item in classified_text]
            df['score'] = [item['score'] * 100 for item in classified_text]
    
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
        #    text = text.lower()  # lower case
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

#Text Classification
elif selection == 'Text Classification':
    st.title('Text Classification App')
    st.write("""
    The model used for this classification is **typeform/distilbert-base-uncased-mnli**. Ensure that you have done good groundwork in
    identifying the recurring ideas from the text. This classifier takes either single words or short phases that are separated by a comma. \n \n
    **TIPS FOR USAGE:** \n
    First, run the word cloud and the N-GRAM analysis so you can identify the big ideas. You can modify the words / phrases to get the best results. \n
    When breaking the data into fragments, just be aware that the analysis will be based on the fragment and may not be representative of the whole part.
    """)
   # Custom CSS for download buttons
    st.markdown("""
    <style>
    .download-btn {
        background-color: #c48939;
        color: white;
        padding: 14px 20px;
        margin: 8px 0;
        border: none;
        cursor: pointer;
        width: 100%;
        text-align: center;
    }
    .download-btn:hover {
        background-color: #005baa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Define user input for Classification Labels
    labels = st.text_input('Enter your classification labels, separated by comma')
    
    # Split the labels into a list
    labels = [label.strip() for label in labels.split(',')]
    
    # File Upload
    uploaded_file = st.file_uploader(" **Upload CSV or Excel with a column name 'text'. ** ", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.markdown('Uploaded file is empty. Please upload another.')
                raise ValueError("Empty File")
        except Exception as e:
            try:
                df = pd.read_excel(uploaded_file)
                if df.empty:
                    st.markdown('Uploaded file is empty. Please upload another.')
                    raise ValueError("Empty File")
            except Exception as e:
                st.markdown(f'Error reading file: {e}')
                raise
    
        # Drop empty rows based on the 'text' column
        df.dropna(subset=['text'], inplace=True)
    
        if 'text' not in df.columns:
            st.markdown('File does not have a `text` column. Please upload another.')
        elif not labels:
            st.markdown('Please enter some classification labels.')
            # Add type and value checks for labels and text
            if not isinstance(labels, list):
                st.error("Labels should be a list.")
                st.stop()

            if any(not isinstance(text, str) for text in df['text']):
                st.error("All text data should be of type string.")
                st.stop()
        else:
            def classify(text, labels):
                result = classifier(text, labels)
                if 'labels' in result and 'scores' in result:
                    return pd.Series([result['labels'][0], result['scores'][0]], index=['label', 'score'])
                else:
                    return pd.Series([None, None], index=['label', 'score'])

            if len(df) > 0 and labels is not None and len(labels) > 0:
                df[['label', 'score']] = df['text'].apply(lambda x: classify(x, labels))
    
            # Normalize the labels and calculate frequency distribution
            if 'label' in df.columns:
                label_counts = df['label'].value_counts(normalize=True)
            label_counts = label_counts.sort_values(ascending=False)
            
            # Pie chart
            fig, ax = plt.subplots()
            ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=['#005baa', '#c48939', '#d61c33'])
            ax.axis('equal')
            
            # Remove background
            ax.set_facecolor('none')
            fig.patch.set_visible(False)
    
            # Convert the Matplotlib figure to a BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Convert DataFrame to CSV and encode
            csv = df.to_csv(index=False)
            csv_base64 = base64.b64encode(csv.encode()).decode()
    
            # Custom download buttons
            
            st.markdown(f'<a href="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}" download="label_distribution.png" class="download-btn"><i class="fas fa-download"></i> Download Pie Chart</a>', unsafe_allow_html=True)
    
            # Display pie chart
            st.pyplot(fig)

            st.markdown(f'<a href="data:text/csv;base64,{csv_base64}" download="classified_data.csv" class="download-btn"><i class="fas fa-download"></i> Download Data as CSV</a>', unsafe_allow_html=True)
            # Display DataFrame
            st.write(df)
    


    

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
    
            st.write(result_df)
            
elif selection == 'Combined Analysis':
    st.title("Combined Analysis")

    st.subheader("Text Classification")
    
   # Custom CSS for download buttons
    st.markdown("""
    <style>
    .download-btn {
        background-color: #c48939;
        color: white;
        padding: 14px 20px;
        margin: 8px 0;
        border: none;
        cursor: pointer;
        width: 100%;
        text-align: center;
    }
    .download-btn:hover {
        background-color: #005baa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Define user input for Classification Labels
    labels = st.text_input('Enter your classification labels, separated by comma')
    
    # Split the labels into a list
    labels = [label.strip() for label in labels.split(',')]
    
    # File Upload
    uploaded_file = st.file_uploader(" **Upload CSV or Excel with a column name 'text'. ** ", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.empty:
                st.markdown('Uploaded file is empty. Please upload another.')
                raise ValueError("Empty File")
        except Exception as e:
            try:
                df = pd.read_excel(uploaded_file)
                if df.empty:
                    st.markdown('Uploaded file is empty. Please upload another.')
                    raise ValueError("Empty File")
            except Exception as e:
                st.markdown(f'Error reading file: {e}')
                raise
    
        # Drop empty rows based on the 'text' column
        df.dropna(subset=['text'], inplace=True)
    
        if 'text' not in df.columns:
            st.markdown('File does not have a `text` column. Please upload another.')
        elif not labels:
            st.markdown('Please enter some classification labels.')
        else:
            def classify(text, labels):
                result = classifier(text, labels)
                return pd.Series([result['labels'][0], result['scores'][0]], index=['label', 'score'])
            
            df[['label', 'score']] = df['text'].apply(lambda x: classify(x, labels))
    
            # Normalize the labels and calculate frequency distribution
            label_counts = df['label'].value_counts(normalize=True)
            
            # Sort by frequency
            label_counts = label_counts.sort_values(ascending=False)
            
            # Pie chart
            fig, ax = plt.subplots()
            ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=['#005baa', '#c48939', '#d61c33'])
            ax.axis('equal')
            
            # Remove background
            ax.set_facecolor('none')
            fig.patch.set_visible(False)
    
            # Convert the Matplotlib figure to a BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Convert DataFrame to CSV and encode
            csv = df.to_csv(index=False)
            csv_base64 = base64.b64encode(csv.encode()).decode()
    
            # Custom download buttons
            
            st.markdown(f'<a href="data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}" download="label_distribution.png" class="download-btn"><i class="fas fa-download"></i> Download Pie Chart</a>', unsafe_allow_html=True)
    
            # Display pie chart
            st.pyplot(fig)

            st.markdown(f'<a href="data:text/csv;base64,{csv_base64}" download="classified_data.csv" class="download-btn"><i class="fas fa-download"></i> Download Data as CSV</a>', unsafe_allow_html=True)
            # Display DataFrame
            st.write(df)

    st.subheader("Sentiment")
    classifier = pipeline('sentiment-analysis')
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            df = pd.read_excel(uploaded_file)
    
        if 'text' not in df.columns:
            st.markdown('File does not have a `text` column. Please upload another.')
        else:
            # Drop NaN or empty values and ensure the 'text' column contains strings
            df.dropna(subset=['text'], inplace=True)
            df['text'] = df['text'].astype(str)
    
            # Perform Text Classification in a batch
            classified_text = classifier(df['text'].tolist())
            
            df['label'] = [item['label'] for item in classified_text]
            df['score'] = [item['score'] * 100 for item in classified_text]
    
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

    st.subheader("N-GRAMS - Thematic")
       
    ngram_min = st.slider("Minimum N-gram Range", 1, 5, 2)
    ngram_max = st.slider("Maximum N-gram Range", ngram_min, 10, 3)
    
    if uploaded_file is not None:
        df_ngram = thematic_analysis(uploaded_file, ngram_min, ngram_max)
        st.write(df_ngram)
    
        if df_ngram is not None and not df_ngram.empty:
            top_ngrams = df_ngram.head(50)
    
            chart = alt.Chart(top_ngrams).mark_bar().encode(
                y=alt.Y('ngram:O', sort='-x'),
                x='frequency:Q',
                tooltip=['ngram', 'frequency']
            ).properties(
                title='Top 50 N-grams',
                width=600
            )
    
            st.altair_chart(chart)  
    
# Topic Modelling Page
elif selection == 'Text Exploration':
    st.title('Text Exploration')          

        # OpenAI API Key input
    api_key = st.text_input("Enter your OpenAI API Key", type="password")

    # File upload
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file and api_key:
        df = pd.read_excel(uploaded_file)

        # Column selection
        if not df.empty:
            column = st.selectbox('Select the column to analyze', df.columns)

            # Total Rows Count
            st.metric(label="Total Rows", value=df.shape[0])

            # Summary of Responses
            concatenated_text = ' '.join(df[column].astype(str))
            summary = summarize_text(api_key, concatenated_text)
            st.write("Summary of Responses:")
            st.write(summary)

            # Sentiment Analysis and Pie Chart
            df['sentiment'] = df[column].apply(lambda x: analyze_sentiment(api_key, x))
            sentiment_counts = df['sentiment'].value_counts()
            plt.figure(figsize=(8, 8))
            plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
            plt.title('Sentiment Analysis')
            st.pyplot(plt)

            # Frequency Count Bar Graph of Word Count
            word_counts = df[column].str.split().str.len()
            predefined_bins = [0, 5, 10, 15, 25, 50, 100]
            max_bin = max(predefined_bins[-1], max(word_counts) + 1)
            bins = predefined_bins + [max_bin] if max_bin > predefined_bins[-1] else predefined_bins

            binned_data = pd.cut(word_counts, bins=bins, include_lowest=True, right=False)
            frequency_counts = binned_data.value_counts().sort_index()
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=frequency_counts.index.categories, y=frequency_counts.values, palette="viridis")
            ax.set_xlabel('Word Count Ranges')
            ax.set_ylabel('Frequency')
            ax.set_title('Frequency Count of Word Ranges')
            ax.set_xticklabels([f"{int(i.left)}-{int(i.right)-1}" for i in frequency_counts.index.categories])
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')
            st.pyplot(plt)

            # User input for words to exclude from the word cloud
            words_to_exclude = st.text_input("Enter words to exclude from the word cloud, separated by commas").split(',')

            # Word Cloud
            lemmatizer = WordNetLemmatizer()
            words = ' '.join(df[column].fillna('').astype(str)).lower()
            words = ' '.join([lemmatizer.lemmatize(word) for word in words.split() if word not in stopwords.words('english') and word not in words_to_exclude])
            wordcloud = WordCloud(width=800, height=400).generate(words)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

            # Toxicity Detection
            df['toxicity'] = df[column].apply(lambda x: detect_toxicity(api_key, x))
            toxic_responses = df[df['toxicity'] == '2']
            st.write("Toxic Responses:")
            st.table(toxic_responses[[column, 'toxicity']])

            # Analyze for positive aspects
            positive_prompt = "List a maximum of up to 10 positive things. Use major group categories. Be strict to 10, not more:\n\n"
            positive_aspects = analyze_text(api_key, concatenated_text, positive_prompt)
            st.subheader("Observation of posivite aspects")
            st.write(positive_aspects)

            # Analyze for negative aspects
            negative_prompt = "List a maximum of 10 negative things. Use major group categories. Be strict to 10, not more:\n\n"
            negative_aspects = analyze_text(api_key, concatenated_text, negative_prompt)
            st.subheader("Observation of negative aspects")
            st.write(negative_aspects)

            # Analyze for recommendations
            recommendations_prompt = "Give strictly up to 10 recommendations.:\n\n"
            recommendations = analyze_text(api_key, concatenated_text, recommendations_prompt)
            st.subheader("Recommendations")
            st.write(recommendations)
