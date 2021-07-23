
import config
import preprocessors
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# read testing/evaluation data
data_test = pd.read_csv(config.TEST_DATA_FILE)
text_test = data_test[config.FEATURES]
type_test = data_test[config.TARGET]

text_test = text_test.apply(preprocessors.remove_punc)
text_test = text_test.apply(preprocessors.preprocess)
text_test = text_test.apply(preprocessors.stem)
text_test = text_test.apply(preprocessors.lemma)
text_test = text_test.apply(lambda x: ' '.join(x))

# Bag of Words model
bow = joblib.load("bag_of_words")
text_test_bow = bow.transform(text_test)
# Using TFIDF (Term Frequency Inverse Document Frequency)
tfidf = joblib.load('tfidf')
text_test_tfidf = tfidf.transform(text_test_bow)

X_test = text_test_tfidf

y_test = type_test.values
le = joblib.load('labelencoder')
y_test = le.transform(y_test)

msg_classifier = joblib.load("SVM_msg_classifier")
y_pred = msg_classifier.predict(X_test)

print('--------------------------------------------------------------------------')
# Accuracy score for the testset
print()
print("Accuracy Score for the Testset is {}".format(accuracy_score(y_test, y_pred)))
print()
# Classification report for the testset
print('Classification Report for the Testset: ')
print()
print(classification_report(y_test, y_pred))








