import streamlit as st
import pandas as pd
import altair as alt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# Download the NLTK stopwords list if not downloaded
import nltk
nltk.download('stopwords')

# Function to download data as a CSV file
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

# Streamlit App
st.title("Thematic Analysis Using N-Grams")
uploaded_file = st.file_uploader("Choose an Excel file containing 'text' column", type="xlsx")

ngram_min = st.slider("Minimum N-gram Range", 1, 5, 2)
ngram_max = st.slider("Maximum N-gram Range", ngram_min, 5, 4)

if uploaded_file is not None:
    df_ngram = thematic_analysis(uploaded_file, ngram_min, ngram_max)
    st.write(df_ngram)

    if df_ngram is not None and not df_ngram.empty:
        top_ngrams = df_ngram.head(15)

        chart = alt.Chart(top_ngrams).mark_bar().encode(
            y=alt.Y('ngram:O', sort='-x'),
            x='frequency:Q',
            tooltip=['ngram', 'frequency']
        ).properties(
            title='Top 15 N-grams',
            width=600
        )

        st.altair_chart(chart)
         # Option to download the chart
        st.markdown("""
            <a href="javascript:downloadSvg()">Download Chart as PNG</a>
            <script>
                function downloadSvg() {
                    var svgElement = document.querySelector('svg');
                    var svgString = new XMLSerializer().serializeToString(svgElement);
                    var canvas = document.createElement('canvas');
                    var ctx = canvas.getContext('2d');
                    var DOMURL = self.URL || self.webkitURL || self;
                    var img = new Image();
                    var svgBlob = new Blob([svgString], {type: 'image/svg+xml;charset=utf-8'});
                    var url = DOMURL.createObjectURL(svgBlob);
                    img.onload = function() {
                        ctx.drawImage(img, 0, 0);
                        var png = canvas.toDataURL('image/png');
                        var dlLink = document.createElement('a');
                        dlLink.download = 'chart.png';
                        dlLink.href = png;
                        dlLink.dataset.downloadurl = ['image/png', dlLink.download, dlLink.href].join(':');
                        document.body.appendChild(dlLink);
                        dlLink.click();
                        document.body.removeChild(dlLink);
                    };
                    img.src = url;
                }
            </script>
        """, unsafe_allow_html=True)
