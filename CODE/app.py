
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
app = Flask(__name__)
app.config['upload folder']='uploads'


@app.route('/')
def home():
    return render_template('index.html')
global path

@app.route('/load data',methods=['POST','GET'])
def load_data():
    if request.method == 'POST':

        file = request.files['file']
        filetype = os.path.splitext(file.filename)[1]
        if filetype == '.csv':
            path = os.path.join(app.config['upload folder'], file.filename)
            file.save(path)
            print(path)
            return render_template('load data.html',msg = 'success')
        elif filetype != '.csv':
            return render_template('load data.html',msg = 'invalid')
        return render_template('load data.html')
    return render_template('load data.html')


@app.route('/view data',methods = ['POST','GET'])
def view_data():
    global df,data2
    file = os.listdir(app.config['upload folder'])
    path = os.path.join(app.config['upload folder'],file[0])

    df = pd.read_csv(path)
    data = df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)
    data1 = data.replace('?', np.nan)

    print(data1)

    data2 = data1.fillna(method="ffill")
    data2 = data2.astype(float)
    print(data2)
    print(df)
    return render_template('view data.html',col_name =data2.columns.values,row_val = list(data2.values.tolist()))

@app.route('/model',methods = ['POST','GET'])
def model():
    if request.method == 'POST':
        global scores1,scores2,scores3,scores4,scores5
        global df
        filename = os.listdir(app.config['upload folder'])
        path = os.path.join(app.config['upload folder'],filename[0])
        df = pd.read_csv(path)
        global testsize

        testsize =int(request.form['testing'])
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

        

        model = int(request.form['selected'])
        if model == 1:
            lr = LogisticRegression()
            model1 = lr.fit(x_train,y_train)
            pred1 = model1.predict(x_test)

            scores1 = accuracy_score(y_test,pred1)


            return render_template('model.html',score = round(scores1,4),msg = 'accuracy',selected  = 'Logistic Regression')
        elif model == 2:
            rfc = RandomForestClassifier(max_depth=5, n_estimators=5,random_state=0)
            model2 = rfc.fit(x_train,y_train)
            pred2 = model2.predict(x_test)
            scores2 =accuracy_score(y_test,pred2)
            return render_template('model.html',msg = 'accuracy',score = round(scores2,3),selected = 'RANDOM FOREST CLASSIFIER')
        elif model == 3:
            ex= ExtraTreeClassifier()
            model3 = ex.fit(x_train,y_train)
            pred3 = model3.predict(x_test)
            scores3 = accuracy_score(y_test,pred3)
            return render_template('model.html',msg = 'accuracy',score = round(scores3,3),selected = 'ExtraTreeClassifier')
        elif model == 4:
            cb = CatBoostClassifier()
            model4 = cb.fit(x_train,y_train)
            pred4 = model4.predict(x_test)
            scores4 = accuracy_score(y_test,pred4)
            return render_template('model.html',msg = 'accuracy',score = round(scores4,3),selected = 'CatBoostClassifier')
        elif model == 5:
            svc = SVC(C=5)
            model5 = svc.fit(x_train,y_train)
            pred5 = model5.predict(x_test)
            scores5 = accuracy_score(y_test,pred5)
            return render_template('model.html',msg = 'accuracy',score = round(scores5,3),selected = 'SVC')


    return render_template('model.html')

@app.route('/prediction',methods = ['POST',"GET"])
def prediction():
    if request.method == 'POST':

        a = float(request.form['a'])
        b = float(request.form['b'])
        c = float(request.form['c'])
        d = float(request.form['d'])
        e = float(request.form['e'])
        f = float(request.form['f'])
        g = float(request.form['g'])
        h = float(request.form['h'])
        i = float(request.form['i'])
        j = float(request.form['j'])
        k = float(request.form['k'])
        l = float(request.form['l'])

        values = [[float(a),float(b),float(c),float(d),float(e),float(f),float(g),float(h),float(i),float(j),float(k),float(l)]]
        n111 = np.array(values)

        dtc = CatBoostClassifier()
        model = dtc.fit(x_train,y_train)

        pred = model.predict(n111.reshape(1,-1))
        print(pred)
        type(pred)

        if pred == [1]:
            msg = "The Predicted Output Is Having Cervical Cancer Symptoms"
        elif pred == [0]:
            msg = "The Predicted Output Is Normal Symptoms"

        return render_template('prediction.html',msg =msg)
    return render_template('prediction.html')

@app.route("/graph",methods=['GET','POST'])
def graph():
    model_list = ["Linear Regression","Random Forest Classifier","Extra Tree Classifier","Cat Boost Classifier","Support Vector Classification"]
    i = [scores1,scores2,scores3,scores4,scores5]
    max_score = max(i)
    max_accured_index = i.index(max_score)
    
    msg = model_list[max_accured_index]+" has more accuracy of "+str(round(max_score*100,4))+"%"
    
    return render_template('graph.html',i=i,msg = msg)

if __name__ == '__main__':
    app.run(debug=True)