
import config
import preprocessors
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# read training data
data_train = pd.read_csv(config.TRAINING_DATA_FILE)
text_train = data_train[config.FEATURES]
type_train = data_train[config.TARGET]

text_train = text_train.apply(preprocessors.remove_punc)
text_train = text_train.apply(preprocessors.preprocess)
text_train = text_train.apply(preprocessors.stem)
text_train = text_train.apply(preprocessors.lemma)
text_train = text_train.apply(lambda x: ' '.join(x))

# Bag of Words model
bow = CountVectorizer().fit(text_train)
joblib.dump(bow, "bag_of_words")
message_bow = bow.transform(text_train)
# Using TFIDF (Term Frequency Inverse Document Frequency)
tfidf = TfidfTransformer().fit(message_bow)
joblib.dump(tfidf, "tfidf")
message_tfidf = tfidf.transform(message_bow)

X_train = message_tfidf

y_train = type_train.values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
joblib.dump(le, 'labelencoder')

# Applying SVM (Support Vector Machine) algorithm

msg_classifier = SVC(kernel='rbf')

print("Training is started............")
msg_classifier.fit(X_train, y_train)
print("Training is completed..........")
print()

joblib.dump(msg_classifier, "SVM_msg_classifier")

y_train_pred = msg_classifier.predict(X_train)

print('--------------------------------------------------------------------------')
# Accuracy score for the training set
print("Accuracy Score for the Training Set is {}".format(accuracy_score(y_train, y_train_pred)))

print()
# Classification report for the trainingset
print("Classification Report for the Training Set: \n")
print()
print(classification_report(y_train, y_train_pred))





















