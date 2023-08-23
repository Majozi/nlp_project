import streamlit as st

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

# Text Classification Page
elif selection == 'Text Classification':
    st.title('Text Classification')
    def classification():
    candidate_labels_input = st.text_input("Enter candidate labels, separated by commas (e.g., positive,negative):")
    candidate_labels = candidate_labels_input.split(",") if candidate_labels_input else []

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df = df[['text']].dropna()

        if not candidate_labels:
            candidate_labels = ["positive", "negative"]

        classifier = pipeline(task="zero-shot-classification", model="facebook/bart-large-mnli")

        res = classifier(df['text'].tolist(), candidate_labels=candidate_labels)

        # Assuming you have the final DataFrames ready, you can display them as tables:
        st.write(classified_text_neg_pos_sample)
        st.write(classified_text_normalized_neg_pos)
        st.write(bert_classification_sample)

        # You can also provide download links for the Excel files:
        # ...

if __name__ == '__main__':
    classification()
    st.write("""
    Text classification is the task of categorizing text into predefined classes or labels. 
    It includes applications like spam detection, topic labeling, sentiment analysis, and more. 
    Machine learning models, especially large language models, have excelled in this domain.
    """)

# Topic Modelling Page
elif selection == 'Topic Modelling':
    st.title('Topic Modelling')
    st.write("""
    Topic modeling is a technique to discover the hidden thematic structure in a large collection of documents.
    It's incredibly useful in organizing, understanding and summarizing large datasets of textual information.
    Algorithms like LDA (Latent Dirichlet Allocation) are commonly used for this purpose.
    """)
