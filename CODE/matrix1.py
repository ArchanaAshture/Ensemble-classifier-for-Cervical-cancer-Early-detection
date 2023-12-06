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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
path ="../DATA SET/kag_risk_factors_cervical_cancer.csv"
df = pd.read_csv(path)
data = df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)
data1 = data.replace('?', np.nan)

print(data1)

data2 = data1.fillna(method="ffill")
data2 = data2.astype(float)
print(data2)
print(df)

global scores1,scores2,scores3,scores4,scores5
df = pd.read_csv(path)
global testsize

testsize =30

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


lr = LogisticRegression()
model1 = lr.fit(x_train,y_train)
pred1 = model1.predict(x_test)

scores1 = accuracy_score(y_test,pred1)

cm1 = confusion_matrix(y_test, pred1)
plot_confusion_matrix(model1, x_test, y_test)
plt.title("Logistic Regression")
plt.show()


rfc = RandomForestClassifier(max_depth=5, n_estimators=5,random_state=0)
model2 = rfc.fit(x_train,y_train)
pred2 = model2.predict(x_test)
scores2 =accuracy_score(y_test,pred2)

cm2 = confusion_matrix(y_test, pred2)
plot_confusion_matrix(model2, x_test, y_test)
plt.title("Random Forest Classifier")
plt.show()


ex= ExtraTreeClassifier()
model3 = ex.fit(x_train,y_train)
pred3 = model3.predict(x_test)
scores3 = accuracy_score(y_test,pred3)

cm3 = confusion_matrix(y_test, pred3)
plot_confusion_matrix(model3, x_test, y_test)
plt.title("ExtraTreeClassifier")
plt.show()


cb = CatBoostClassifier()
model4 = cb.fit(x_train,y_train)
pred4 = model4.predict(x_test)
scores4 = accuracy_score(y_test,pred4)
cm3 = confusion_matrix(y_test, pred4)
plot_confusion_matrix(model4, x_test, y_test)
plt.title("CatBoost")
plt.show()


svc = SVC(C=5)
model5 = svc.fit(x_train,y_train)
pred5 = model5.predict(x_test)
scores5 = accuracy_score(y_test,pred5)
cm3 = confusion_matrix(y_test, pred5)
plot_confusion_matrix(model5, x_test, y_test)
plt.title("SVC")
plt.show()


fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

plot_confusion_matrix(model1, x_test, y_test, ax=axs[0,0])
axs[0,0].set_title("Logistic Regression")

plot_confusion_matrix(model2, x_test, y_test, ax=axs[0,1])
axs[0,1].set_title("Random Forest Classifier")

plot_confusion_matrix(model3, x_test, y_test, ax=axs[0,2])
axs[0,2].set_title("ExtraTreeClassifier")

plot_confusion_matrix(model4, x_test, y_test, ax=axs[1,0])
axs[1,0].set_title("CatBoost")

plot_confusion_matrix(model5, x_test, y_test, ax=axs[1,1])
axs[1,1].set_title("SVC")

plt.tight_layout()
plt.show()
