import streamlit as st
st.title('Sentiment Analysis')
import pandas as pd
df = pd.read_table('/content/Reviews.tsv')
x = df.iloc[:,1].values
y = df.iloc[:,0].values
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
text_model = Pipeline([('tfidf',TfidfVectorizer()),('model',SVC())])
text_model.fit(x,y)
select = st.text_input('Type your Message')
op = text_model.predict([select])
st.title(op[0])
