from flask import Flask, render_template, request, session, url_for, redirect, flash, send_from_directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report, accuracy_score
import mysql.connector
import joblib
from flask_login import login_user, current_user, logout_user, login_required
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from pprint import pprint
import os
from sklearn.model_selection import KFold

app = Flask(__name__)
app.secret_key = 'zidanzulkhairyan'
app.config['UPLOAD_FOLDER'] = 'Uploaded Files'

ALLOWED_EXTENSIONS = {'xlsx','csv'}


mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="knn"
)

mycursor = mydb.cursor()

user = {
    'username': 'admin',
    'password': 'admin'
}

@app.route('/', methods=['GET', 'POST'])
def login():
    # if current_user.is_authenticated:
    #     return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == user['username'] and password == user['password']:
            return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
# @login_required
def index():
    mycursor.execute('SELECT * FROM arsip')
    myresult = mycursor.fetchall()
    return render_template('index.html', myresult=myresult)

@app.route('/hapus', methods=['GET', 'POST'])
def hapus():
    mycursor = mydb.cursor()

    truncate = "TRUNCATE TABLE arsip"

    mycursor.execute(truncate)

    hapus = "DELETE FROM arsip"

    mycursor.execute(hapus)

    mydb.commit()
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        file = request.files['upload-file']
        if file.filename == '':
            flash('tidak ada file yang dipilih')
            return redirect(url_for('index'))
        if file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
            flash('file harus dalam format xlsx dan csv!')
            return redirect(url_for('index'))
        
        dfup = pd.read_excel(file)

        for index, row in dfup.iterrows():
            sql = "INSERT INTO data_training (nomor_surat, judul_surat, asal_surat, tujuan_surat, keterangan, file_upload) VALUES (%s,%s,%s,%s,%s,%s)"
            val = (row['Nomor Surat'], row['Judul Surat'], row['Asal Surat'], row['Tujuan Surat'],row['Keterangan'], row['File'])
            mycursor.execute(sql, val)
        mydb.commit()
        return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/prediksi', methods=['GET','POST'])
def prediksi():
    mycursor.execute('SELECT * FROM arsip')
    myresult = mycursor.fetchall()
    data = pd.DataFrame(myresult)
    data = data.rename(columns={0: 'ID', 1: 'Nomor Surat', 2: 'Judul Surat', 3: 'Asal Surat', 4: 'Tujuan Surat'
                                , 5: 'Keterangan', 6: 'File'})

    atribut = data['Judul Surat']
    target = data['Keterangan']

    X_train, X_test, y_train, y_test = train_test_split(atribut,target,test_size=0.4)
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', KNeighborsClassifier(n_neighbors=5)),
                         ])

    #joblib.dump(text_clf, 'model_nb.pkl')

    score, precision, recall, f1 = [],[],[],[]

    # predicted = text_clf.predict(X_test)  

    score, precision, recall, f1 = [], [], [], []

    kf = KFold(n_splits=10)

    for train_index, test_index in kf.split(data):
        X_train = data.iloc[train_index]['Judul Surat']
        X_test = data.iloc[test_index]['Judul Surat']
        y_train = data.iloc[train_index]['Keterangan']
        y_test = data.iloc[test_index]['Keterangan']

        text_clf.fit(X_train, y_train)
        predicted = text_clf.predict(X_test)

        score.append(accuracy_score(predicted,y_test))
        precision.append(precision_score(predicted,y_test,average='weighted'))
        recall.append(recall_score(predicted,y_test, average='weighted'))
        f1.append(f1_score(predicted,y_test,average='weighted'))

    dft = pd.DataFrame(score)
    dft.columns = ['Akurasi']
    dft['Precision'] = precision
    dft['Recall'] = recall
    dft['F1-Score'] = f1
    dft = dft.round(2)
    dft *= 100

    d1 = pd.DataFrame(text_clf['vect'].get_feature_names_out())
    d2 = pd.DataFrame(text_clf['tfidf'].idf_) 
    datapre = pd.concat([d1, d2], axis=1)

    return render_template(
        'prediksi.html',
        meanA=dft['Akurasi'].mean(),
        meanP=dft['Precision'].mean(),
        meanR=dft['Recall'].mean(),
        meanF=dft['F1-Score'].mean(),
        heading=dft.columns,
        dfup=dft.to_numpy(),
        dataheading=data.columns,
        data=data.to_numpy(),
        datapreval = datapre.to_numpy()
    )


@app.route('/inputprediksi', methods=['GET', 'POST'])
def inputprediksi():
    mycursor.execute('SELECT * FROM data_training')
    mypred = mycursor.fetchall()
    datatrain = pd.DataFrame(mypred)
    datatrain = datatrain.rename(columns={0: 'ID', 1: 'Nomor Surat', 2: 'Judul Surat', 3: 'Asal Surat', 4: 'Tujuan Surat'
                                , 5: 'Keterangan', 6: 'File'})


    if request.method == 'POST':
        nosu = request.form['nomor_surat']
        jusu = request.form['judul_surat']
        asal = request.form['asal_surat']
        tujuan = request.form['tujuan_surat']
        upfile = request.files ['file_upload']

        formdata = {'Nomor Surat': [nosu], 'Judul Surat': [jusu], 'Asal Surat': [asal], 'Tujuan Surat': [tujuan], 'File': [upfile.filename]}

        dataasal = pd.DataFrame(formdata)
        data = pd.DataFrame(formdata)
        datapredict = data.copy()
        datapredict = datapredict.rename(columns={0: 'ID', 1: 'Nomor Surat', 2: 'Judul Surat', 3: 'Asal Surat', 4: 'Tujuan Surat'
                                , 5: 'Keterangan', 6: 'File'})

        atribut = datatrain['Judul Surat']
        target = datatrain['Keterangan']

        X_train, X_test, y_train, y_test = train_test_split(atribut,target,test_size=0.2)
        text_clf = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf', KNeighborsClassifier()),
                            ])

        text_clf.fit(X_train,y_train)

        # upload_model = joblib.load('model_bener.pkl')
        # predict = upload_model.predict(datapredict)
        predict = text_clf.predict(datapredict['Judul Surat'])
        datapredict['Keterangan'] = predict
        data['keterangan'] = predict

        # d1 = pd.DataFrame(text_clf['vect'].get_feature_names_out())
        # d2 = pd.DataFrame(text_clf['tfidf'].idf_) 
        # datapre = pd.concat([d1, d2], axis=1)

        for index, row in datapredict.iterrows():
            sql = "INSERT INTO arsip (nomor_surat, judul_surat,asal_surat,tujuan_surat,keterangan,file_upload) VALUES (%s,%s,%s,%s,%s,%s)"
            val = (row['Nomor Surat'], row['Judul Surat'],row['Asal Surat'], row['Tujuan Surat'],row['Keterangan'], row['File'])
            mycursor.execute(sql, val)
        mydb.commit()

        upfile.save(os.path.join(app.config['UPLOAD_FOLDER'], upfile.filename))

        flash(f'Surat Merupakan :{predict}')

        return render_template('inputprediksi.html')
    return render_template("inputprediksi.html")

@app.route('/fileprediksi', methods=['GET', 'POST'])
def fileprediksi():

    if request.method == 'POST':
        file = request.files['upload-file']
        if file.filename == '':
            flash('tidak ada file yang dipilih')
            return redirect(url_for('fileprediksi'))
        if file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
            flash('file harus dalam format xlsx dan csv!')
            return redirect(url_for('fileprediksi'))
        datafile = pd.read_excel(file)

        datalama = datafile.copy()
        datapredict = datafile.copy()

        # upload_model = joblib.load('model_fitted.pkl')
        # predict = upload_model.predict(datapredict)

        # datapredict['Keterangan'] = predict

        # datafile['Keterangan'] = predict

    

        for index, row in datafile.iterrows():
            sql = "INSERT INTO arsip (nomor_surat, judul_surat,asal_surat,tujuan_surat,keterangan,file_upload) VALUES (%s,%s,%s,%s,%s,%s)"
            val = (row['Nomor Surat'], row['Judul Surat'],row['Asal Surat'], row['Tujuan Surat'], row['Keterangan'], row['File'])
            mycursor.execute(sql, val)
        mydb.commit()

        return render_template('fileprediksi.html', datafile=datapredict.to_numpy(), datalama=datalama.to_numpy())
        
    return render_template('fileprediksi.html')

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    #  upfile.filename
    return send_from_directory(app.config['UPLOAD_FOLDER'], path=filename+'.pdf', as_attachment=True)

@app.route("/logout")
def logout():
    return redirect(url_for("login"))

if __name__ == '__main__':
    app.run(debug=True)