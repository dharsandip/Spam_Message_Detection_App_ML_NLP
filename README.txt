This is a complete machine learning end-to-end project for building a Web App for the detection of Spam messages or the classification of any messages into Spam and Ham with very high accuracy. For this project, NLP Tools (NLTK, Gensim etc.), Bag of words model, TFIDF and classification model (SVM) are used. For building the Web App, streamlit library is used. We productionized the enire machine learning pipeline, created Web API with front end UI in streamlit, created docker image of the enire application and at the end deployed the containerized ML application onto Google Kubernetes Engine (GKE) in GCP and exposed it to internet and finally tested this ML App from browsers and predicted spam successfully for a single text message (just typed in the box in App) and also multiple different text messages (read from a csv file). 
We tried Multinomial Naïve Bayes, SVM and Random Forest algorithms for this classification problem and finalized with SVM since it gave the best results. 


Results:

------------------------------------------------------------
Accuracy Score for the Training Set is 0.9969238656754679

Classification Report for the Training Set:


              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3377
           1       1.00      0.98      0.99       524

    accuracy                           1.00      3901
   macro avg       1.00      0.99      0.99      3901
weighted avg       1.00      1.00      1.00      3901


--------------------------------------------------------

Accuracy Score for the Testset is 0.9760908547519426

Classification Report for the Testset:

              precision    recall  f1-score   support

           0       0.97      1.00      0.99      1450
           1       0.99      0.83      0.90       223

    accuracy                           0.98      1673
   macro avg       0.98      0.91      0.94      1673
weighted avg       0.98      0.98      0.98      1673



