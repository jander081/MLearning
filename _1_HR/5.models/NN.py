# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 12:47:36 2018

@author: JANDER33
"""

#PATHS----------------------------------------------------------------



#MODULES-----------------------------------------------------------

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, initializers, regularizers

import keras.backend as K
from keras.callbacks import EarlyStopping

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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



# NN MODEL----------------------------------------------------------

# baseline model
def NN():
	
    model = Sequential()
    model.add(Dense(44, input_dim=44, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
     
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=NN, epochs=40, batch_size=32, verbose=1) 
                            #callbacks=[EarlyStopping(monitor = 'loss', patience = 1)])

estimator.fit(X_train, y_train)

y_pred = estimator.predict(X_test)
y_logloss = estimator.predict_proba(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
print('log-loss: ', log_loss(y_test, y_logloss))



#KFOLD---------------------------------------------------------------
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

results = cross_val_score(estimator, df, np.ravel(y), cv=kfold)

print("Results: ", (results.mean()*100, results.std()*100))






















