import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk

def thematic():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        # Assuming the file is in Excel format
        df = pd.read_excel(uploaded_file)
        df = df[['text']].dropna()

        stoplist = stopwords.words('english')
        c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(3,4))
        ngrams = c_vec.fit_transform(df['text'])
        count_values = ngrams.toarray().sum(axis=0)
        vocab = c_vec.vocabulary_

        df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()],
                                        reverse=True)).rename(columns={0: 'frequency', 1:'bigram/trigram'})
        
        # You can add some filtering or visualization here if needed
        
        # Download the result as an Excel file
        excel_file = df_ngram.to_excel(index=False)
        b64 = base64.b64encode(excel_file.encode()).decode()  
        href = f'<a href="data:file/excel;base64,{b64}" download="nlp_analysis.xlsx">Download the NLP Analysis</a>'
        st.markdown(href, unsafe_allow_html=True)

if __name__ == '__main__':
    thematic()
