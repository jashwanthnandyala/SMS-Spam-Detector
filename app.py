import streamlit as st
import pickle

import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

nltk.download('stopwords')
nltk.download('punkt')

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('spam.pkl','rb'))
st.title('SMS-spam Detector')

input_sms=st.text_area("Enter the SMS ")

def text_transform(text):
  text=text.lower()
  text=nltk.word_tokenize(text)

  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text=y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text=y[:]
  y.clear()

  for i in text:
     y.append(ps.stem(i))
  text=y[:]
  y.clear()
  return " ".join(text)


if st.button('submit'):
    transformed_sms = text_transform(input_sms)

    vector = tfidf.transform([transformed_sms])

    prediction = model.predict(vector)

    if prediction == 1:
        st.text("Spam!")
    else:
        st.text("Not Spam")

