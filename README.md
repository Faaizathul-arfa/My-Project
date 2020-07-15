# Fake news Detection
 Batch 2.
 
 By : Faaizathul Arfa
 
 
 ![Image](https://www.arabianbusiness.com/public/styles/full_img/public/images/2018/05/18/fake-news.jpg?itok=2c5POe8V)

## Problem Defination:

Developing a machine learning project to distinguish fake news from a real one.we can use supervised learning to implement the model.Using sklearn, building a TfidfVectorizer on our dataset. Then,initializing a Passive Aggressive Classifier,Naive bayes,logistic regresssion,SVM and fit the model. In the end, the accuracy score and the confusion matrix tell us how well the model fares.
 
## Introduction:

The topic of fake news detection on social media has recently attracted tremendous attention. The basic countermeasure of comparing websites against a list of labeled fake news sources is inflexible, and so a machine learning approach is desirable. Our project aims to use detect fake news directly, based on the text content of news articles


 ![Image](https://www.pantechsolutions.net/media/wysiwyg/ML/twitter_fake_news_3.jpg)
 
## Steps involved:
 
 1. Analysing the dataset and cleanig the data.
 
 2. Data preparing.
 
 3. Building the machine learning models.
 
 4. evaluating the models for better accuracy.
 
### 1. Analysing the dataset and cleanig the data

Dataset is in filename.csv form is read and displayed to get the idea of labels .first few lines of the file are displayed 
```
df=pd.read_csv('/content/news.csv')
df.head()
```
```
labels=df.label
print(labels.head())
```
### 2.Data preparation
splitting dataset arrays into two subsets.that is train and test set.This for training the model and test the accuracy.The test size is taken as 0.33

```
from sklearn.model_selection import train_test_split
X=df['text']
y=labels
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33)
```
Initializing the TfidfVectorizer with stop words with maximum document frequency 0.8  .stopwords are the unneccasery words that needs to be filtered before processing natural langauage data.The job of TfidfVectorizer is to turn a collection of raw data into a TF-IDF matrix

next, fit and transform the vectorizer on the train set and transform the vectorizer on the test set 
```
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.8)
tfidf_train=tfidf_vectorizer.fit_transform(X_train) 
tfidf_test=tfidf_vectorizer.transform(X_test)
```


### models used for classification:
**1.passive Agressive Classifier**

The algorithm used for classification here PassiveAgressiveClassifier .Initializing the PassiveAgressiveClassifier and then fit this to tfidf_train and y_train.
```
pac = PassiveAggressiveClassifier(50) 
pac.fit(tfidf_train, y_train)
pred1 = pac.predict(tfidf_test)
```
printing the confusion matrix obtained by above model
```
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,pred1))
```
and determing the acccuracy as below
```
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred1))
```
similarly getting the classification report
```
from sklearn.metrics import classification_report
print(classification_report(y_test,pred1))
```
**2.Logistic Regression**

Logistic regression is another technique borrowed by machine learning from the field of statistics. It is the method for binary classification problems (problems with two class values).Here we initialize the logisitic regression and then fit this to tfidf_train and y_train.

```
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(tfidf_train,y_train)
pred3=lr.predict(tfidf_test)
```
```
print(accuracy_score(y_test,pred3))
print(confusion_matrix(y_test,pred3))
print(classification_report(y_test,pred3))
```

**3.Naive Bayes Classifier**

Naive Bayes classifier is standard Algorithm for classification for text-based data science projects.Naïve Bayes is a conditional probability model which can be used for labeling.The Naïve Bayes rule is based on the theorem formulated by Bayes.
            
            
```
nb_classifier = MultinomialNB(30)
nb_classifier.fit(tfidf_train, y_train)
pred2 = nb_classifier.predict(tfidf_test)
```
Following the same as above to get the confusion matrix ,accuracy and classification report
```
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,pred2))
```
```
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred2))
```
```
from sklearn.metrics import classification_report
print(classification_report(y_test,pred2))
```
**4.support vector machine model**

SVM is a supervised machine learning algorithm which can be used for classification or regression problems. It uses a technique called the kernel trick to transform your data and then based on these transformations it finds an optimal boundary between the possible outputs.
initialize the logisitic regression and then fit this to tfidf_train and y_train.
```
from sklearn.svm import SVC
clf=SVC()
clf.fit(tfidf_train,y_train)
pred4=clf.predict(tfidf_test)
```
```
print(accuracy_score(y_test,pred4))
print(confusion_matrix(y_test,pred4))
print(classification_report(y_test,pred4))
```
## 4. evaluating the models for better accuracy.

 I got highest accuracy in the passive agressive classifier out of the above foue models that is 0.936





