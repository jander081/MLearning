# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:51:20 2018

@author: JANDER33
"""

#PATHS----------------------------------------------------------------



#MODULES-----------------------------------------------------------

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.model_selection import train_test_split, cross_val_score





 # fix random seed for reproducibility
seed = 81
np.random.seed(seed)





#IMPORT------------------------------------------------------------

df = pd.read_csv(r'C:\Users\jander33\Desktop\projects\project5\models\data_incoming\7_17_df_model.csv')

y = df[['TRV.Win']]
df.drop(['TRV.Win'], axis=1, inplace=True)


#SHUFFLED------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.20)



#LOGISTIC-------------------------------------------------------------
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(random_state=seed)
logmodel.fit(X_train,y_train)

y_pred = logmodel.predict(X_test)
y_logloss = logmodel.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
print('log-loss: ', log_loss(y_test, y_logloss))


#logmodel = LogisticRegression()
#logmodel.fit(df, np.ravel(y))

scores = cross_val_score(logmodel, df, np.ravel(y), cv=5)
print(scores)  

#NAIVE BAYES--------------------------------------------------
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
y_logloss = clf.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
print('log-loss: ', log_loss(y_test, y_logloss))


scores = cross_val_score(clf, df, np.ravel(y), cv=5)
print(scores)  



#KNN--------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_logloss = clf.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
print('log-loss: ', log_loss(y_test, y_logloss))


scores = cross_val_score(clf, df, np.ravel(y), cv=5)
print(scores)  


#SVM-------------------------------------------------

from sklearn.svm import SVC
clf = SVC(kernel = 'rbf', random_state = 81)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_logloss = clf.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
print('log-loss: ', log_loss(y_test, y_logloss))


scores = cross_val_score(clf, df, np.ravel(y), cv=5)
print(scores)  














