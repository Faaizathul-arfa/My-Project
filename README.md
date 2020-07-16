# Fake News Detection
 Batch 2.
 
 By : Faaizathul Arfa
 ## overview
 Machine learning Based Minor Project, which uses various classification Algorithms to classify the news into FAKE/REAL, on the basis of their Title and Body-Content. Data  uses algorithms like PassiveAgressiveClassifier,Naives Bayes,Logistic Regression and SVM. It gave 93% accuracy.
 
 
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

 I got highest accuracy in the passive agressive classifier out of the above four models that is 0.936 as shown below :
 ```
 from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred1))
 ```
 whose output is:
 ```
 0.9368723098995696
 ```
 ```
 from sklearn.metrics import confusion_matrix
 print(confusion_matrix(y_test,pred1))
```
whose output is:
```
[[979  60]
 [ 72 980]]
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
