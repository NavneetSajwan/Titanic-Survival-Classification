# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 00:04:18 2019

@author: NavneetSajwan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("train.csv")
#algorithm to check for missing data
for i in df.keys():
    if df.get(i).isnull().values.any():
        print(i,"has missing data")

#taking care of mising data
del df["Cabin"]
del df["Ticket"]
df=df.loc[df.Embarked.isnull()==False]
df['Age']=df.Age.fillna(df.Age.median())

# Slicing x and y
df_X=df.iloc[:,2:]
del df_X['Name']
X=df_X.values
y=df.loc[:]['Survived'].values

#encoding the categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,1]=labelencoder_X.fit_transform(X[:,1])
X[:,6]=labelencoder_X.fit_transform(X[:,6])
onehotencoder=OneHotEncoder(categorical_features=[1,6])
X=onehotencoder.fit_transform(X).toarray()

# Splitting the data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#creating the logistic regression classification model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

#creating the svm classifier 
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',gamma=.15,random_state=0)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

#grid search cv
from sklearn.model_selection import GridSearchCV
parameters=[{'C':[.5,1,1.5],'kernel':['rbf'],'gamma':[.14,.15,.16,.17]}]
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)
grid_search=grid_search.fit(X_train,y_train)
best_accuracy=grid_search.best_score_
best_parameters=grid_search.best_params_
y_pred=grid_search.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
ein=accuracy_score(y_train,grid_search.predict(X_train))
eout=accuracy_score(y_test,y_pred)
#ein calculation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)
accuracies.mean()




