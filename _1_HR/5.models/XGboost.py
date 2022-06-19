# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 06:22:41 2018

@author: jander33

PROJECT 5
"""
#PATHS----------------------------------------------------------------
import sys
sys.path.append(r'C:\Miniconda3\envs\tf\Lib\site-packages')


#MODULES-----------------------------------------------------------
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.model_selection import train_test_split, cross_val_score


from xgboost import XGBClassifier

seed = 81
np.random.seed(seed)


#IMPORT------------------------------------------------------------

df = pd.read_csv(r'C:\Users\jander33\Desktop\projects\project5\models\data_incoming\7_17_df_model.csv')

y = df[['TRV.Win']]
df.drop(['TRV.Win'], axis=1, inplace=True)






#SHUFFLED------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.20)

model = XGBClassifier(max_depth=5, min_child_weight=20,
                      colsample_bytree=0.7, gamma=0.1,
                      reg_alpha=0.1, reg_lambda=0.1,
                      max_delta_step=1, seed=seed)
 
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_logloss = model.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
print('log-loss: ', log_loss(y_test, y_logloss))



# 5-fold CROSS VALIDATION-----------------------------------------------------------

model = XGBClassifier(max_depth=5, min_child_weight=20,
                      colsample_bytree=0.7, gamma=0.1,
                      reg_alpha=0.1, reg_lambda=0.1,
                      max_delta_step=1)
 
model.fit(df, np.ravel(y))


scores = cross_val_score(model, df, np.ravel(y), cv=5)
print(scores)  
















