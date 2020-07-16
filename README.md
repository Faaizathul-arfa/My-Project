# Fake News Detection
 Batch 2.
 
 By : Faaizathul Arfa
 
 
## Overview:
 
Machine learning Based Minor Project, which uses various classification Algorithms to classify the news into FAKE/REAL, on the basis of their Title and Body-Content. Data  uses  algorithms like PassiveAgressiveClassifier,Naives Bayes,Logistic Regression and SVM. It gave 94% accuracy.
 

 <div style="text-align:center"><img src="https://www.arabianbusiness.com/public/styles/full_img/public/images/2018/05/18/fake-news.jpg?itok=2c5POe8V"  width="550" height="350"></div>

## Problem Defination:

Developing a machine learning project to distinguish fake news from a real one.we can use supervised learning to implement the model.Using sklearn, building a TfidfVectorizer on our dataset. Then,initializing a Passive Aggressive Classifier,Naive bayes,logistic regresssion,SVM and fit the model. In the end, the accuracy score and the confusion matrix tells us how well the model fares.
 
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
splitting dataset arrays into two subsets that is train and test set.This for training the model and testing the accuracy.The test size is taken as 0.33

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
We can also use CountVectorizer,that is used to convert a collection of text documents to a vector of term/token counts and then compare withTF-IDF vector to check whether they are same.The code is as follows:
```
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)
```
```
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
print(count_df.head())
```
similarly for tf-idf vector:
```
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
print(tfidf_df.head())
```
and then comparing by checking the no of columns ,however number of columns is same in my case but contents are not that is determined by:
```
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)
```
output:
```
set()
```
Checking both the vectors equal or not:
```
print(count_df.equals(tfidf_df))
```
output:
```
False
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

 I got highest accuracy in the passive agressive classifier out of the above four models that is 0.94 as shown below :
 
 - Accuracy:
  ```
  from sklearn.metrics import accuracy_score
  print(accuracy_score(y_test,pred1))
  ```
  whose output is:
  ```
  0.9426111908177905
  ```
 - Confusion matrix
  ```
  from sklearn.metrics import confusion_matrix
  print(confusion_matrix(y_test,pred1))
  ```
  whose output is:
  ```
  [[1025   44]
   [  76  946]]
  ```
 - To plot confusion matrix:
  ```
  cm1 =metrics.confusion_matrix(y_test, pred1)
  plt.matshow(cm1)
  plt.title('Confusion matrix')
  plt.colorbar()
  plt.show()
  ```
      
   ![Image](data:image/png;base64,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)
