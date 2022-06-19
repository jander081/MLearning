# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:50:26 2018

@author: JANDER33
"""

#ADABOOST

#MODULES-----------------------------------------------------------
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.model_selection import train_test_split, cross_val_score



from adaboost import AdaBoost

seed = 81
np.random.seed(seed)


#IMPORT------------------------------------------------------------

df = pd.read_csv(r'C:\Users\jander33\Desktop\projects\project5\models\data_incoming\7_17_df_model.csv')

y = df[['TRV.Win']]
df.drop(['TRV.Win'], axis=1, inplace=True)

#SHUFFLED------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.20)



clf = AdaBoost(50)

clf.fit(X_train, np.ravel(y_train))

y_pred = clf.predict(X_test)[0]


# test Adaboost

#df = df.iloc[0:20000, ]
#y = np.ravel(y[0:20000])

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))

