

import pandas as  pd

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE
from flask import Flask, render_template, request
import os
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from catboost import CatBoostClassifier

global scores1,scores2,scores3,scores4,scores5
global df
path = "../DATA SET/kag_risk_factors_cervical_cancer.csv"
df = pd.read_csv(path)
global testsize

testsize =30
print(testsize)

global x_train,x_test,y_train,y_test
testsize = testsize/100

data = df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)
data1 = data.replace('?', np.nan)
data2 = data1.fillna(method="ffill")
data2 = data2.astype(float)
X1 = pd.concat([data2['Age'], data2['Number of sexual partners'], data2['First sexual intercourse'],
            data2['Num of pregnancies'], data2['Smokes'], data2['Smokes (packs/year)'],
            data2['Hormonal Contraceptives (years)'], data2['STDs:genital herpes'],
            data2['STDs: Number of diagnosis'], data2['Dx:CIN'], data2['Schiller'], data2['Citology']],
            axis=1)
Y1 = data2['Biopsy']


sm = SMOTE()
X_res, y_res = sm.fit_resample(X1, Y1)
x_train,x_test,y_train,y_test = train_test_split(X_res, y_res,test_size=testsize,random_state=10)

cb = CatBoostClassifier()
model4 = cb.fit(x_train,y_train)
pred4 = model4.predict(x_test)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred4)
cm_normalized = np.round(cm/np.sum(cm,axis=1).reshape(-1,1),2)
sns.heatmap(cm,cmap="Greens",annot=True,cbar_kws={"orientation":"vertical","label":"color bar"},xticklabels=[0,1],yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()