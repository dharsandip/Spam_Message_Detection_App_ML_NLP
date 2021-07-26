
import config
import preprocessors
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import streamlit as st
import os


def spam_detect(text):
    single_text_data = pd.Series(text)
    single_text_data_test = single_text_data.apply(preprocessors.remove_punc)
    single_text_data_test = single_text_data_test.apply(preprocessors.preprocess)
    single_text_data_test = single_text_data_test.apply(preprocessors.stem)
    single_text_data_test = single_text_data_test.apply(preprocessors.lemma)
    single_text_data_test = single_text_data_test.apply(lambda x: ' '.join(x))
    
    bow = joblib.load("bag_of_words")
    single_text_data_test_bow = bow.transform(single_text_data_test)
    tfidf = joblib.load('tfidf')
    single_text_data_test_tfidf  = tfidf.transform(single_text_data_test_bow)
    msg_classifier = joblib.load("SVM_msg_classifier")
    result = msg_classifier.predict(single_text_data_test_tfidf)
    return result

def spam_detect_file(filename):
    data_test = pd.read_csv(filename)
    text_test = data_test[config.FEATURES]
    text_test = text_test.apply(preprocessors.remove_punc)
    text_test = text_test.apply(preprocessors.preprocess)
    text_test = text_test.apply(preprocessors.stem)
    text_test = text_test.apply(preprocessors.lemma)
    text_test = text_test.apply(lambda x: ' '.join(x))
    bow = joblib.load("bag_of_words")
    text_test_bow = bow.transform(text_test)
    tfidf = joblib.load('tfidf')
    text_test_tfidf = tfidf.transform(text_test_bow)
    X_test = text_test_tfidf
    msg_classifier = joblib.load("SVM_msg_classifier")
    result = msg_classifier.predict(X_test)
    return result
    
def main():
    st.title("Spam Message Detection")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Spam Message Detector ML App</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    text_data = st.text_input("Message","Type Here")
    
    result=""
    if st.button("Predict"):
        result = spam_detect(text_data)
        if result == 0:
            st.success('The output is {} and it is ham'.format(result))
        else:
            st.success('The output is {} and it is spam'.format(result))
     
    filename = st.file_uploader("Upload CSV Files",type=['csv'])
    result=""
    if st.button("Predict for messages from the file"):
        result = spam_detect_file(filename)
        st.success('The output is {}'.format(result))
    

if __name__=='__main__':
    main()


