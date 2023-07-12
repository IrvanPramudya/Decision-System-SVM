from sklearn import metrics
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve

# memanggil model
loaded_model = pickle.load(
    open('G:\Skripsi\Projek TA\projek\modelSVMpremierleague', 'rb'))


def prediksiketerangan(baru):
    hasilPrediksi = loaded_model.predict(baru)
    if (hasilPrediksi == 0):
        return'AMF'
    if (hasilPrediksi == 1):
        return'CMF'
    if (hasilPrediksi == 2):
        return'DMF'


st.title(
    'Klasifikasi Midfielder Premier League Menggunakan Algoritma SVM')

df = pd.read_csv(
    "G:\Skripsi\Projek TA\datapl.csv")

if st.sidebar.checkbox('Tampilkan Dataset'):
    st.subheader('Dataset')
    st.write(df)

    df['Nama'] = df['Nama'].str.lower()
    df['Klub'] = df['Klub'].str.lower()
    df['Kebangsaan'] = df['Kebangsaan'].str.lower()

if st.sidebar.checkbox('Ubah Ke huruf kecil'):
    st.subheader('Dataset Menjadi Huruf Kecil')
    st.write(df)

# merubah huruf jadi angka
labelencoder = LabelEncoder()
df['Nama'] = labelencoder.fit_transform(df['Nama'])
df['Klub'] = labelencoder.fit_transform(df['Klub'])
df['Kebangsaan'] = labelencoder.fit_transform(df['Kebangsaan'])
df['Keterangan'] = labelencoder.fit_transform(df['Keterangan'])

if st.sidebar.checkbox('Ubah Huruf Menjadi Angka'):
    st.subheader('Dataset Dari Huruf Menjadi Angka')
    st.write(df)

X = df[['Nama', 'Klub', 'Kebangsaan', 'Minutes Played', 'Passing',
        'Tackles', 'Shots', 'Interception']]  # Atribut yang digunakan

y = df.Keterangan  # Memisah label dengan atribut lain

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

a = df.Keterangan.value_counts()
if st.sidebar.checkbox('Proporsi unik kolom atribut'):
    st.write(a)

clf = svm.SVC(kernel='linear', probability=True)

number = st.sidebar.number_input(
    'Masukan proporsi (contoh : 0.1 berarti 10%)', min_value=0.1, value=0.30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=number)
clf = clf.fit(X_train, y_train)

if st.sidebar.checkbox('Proporsi'):
    st.title('Membagi data train dan test')
    st.subheader('DataTrain Fitur dan Target')
    st.write(X_train, y_train)
    st.subheader('DataTest Fitur dan Target')
    st.write(X_test, y_test)


if st.sidebar.checkbox('Melakukan Pelatihan Model'):
    st.write('Bobot W', clf.coef_)
    st.write('Bias B ', clf.intercept_)
    st.write('Hasil Pelatihan', clf.predict(X_train))
    scores = cross_val_score(clf, X_train, y_train, cv=20)
    st.write(scores)

hasilPrediksi = clf.predict(X_test)
if st.sidebar.checkbox('Prediksi Data Test'):
    st.write('Label Sebenarnya', y_test)
    st.write('Hasil Prediksi', hasilPrediksi)
proba = clf.predict_proba(X_test)

cm = confusion_matrix(y_test, hasilPrediksi)
cm_df = pd.DataFrame(cm,
                     index=['AMF', 'CMF', 'DMF'],
                     columns=['AMF', 'CMF', 'DMF'])

if st.sidebar.button("Show Correlation Plot"):
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("### Heatmap")
    fig, ax = plt.subplots(figsize=(5, 4))
    st.write(sns.heatmap(cm_df, annot=True, fmt='d'))
    st.pyplot()
    # model report
    target = ['AMF', 'CMF', 'DMF']
    st.write(classification_report(
        y_test, hasilPrediksi, target_names=target))

if st.sidebar.checkbox("MSE"):
    st.write('Mean Squared Error')
    st.write(metrics.mean_squared_error(y_test, hasilPrediksi))

if st.sidebar.checkbox("ROC"):
    st.write("ROC")
    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}

    n_class = 3

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, proba[:, i], pos_label=i)

    # plotting
    plt.plot(fpr[0], tpr[0], linestyle='--',
             color='orange', label='Class 0 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--',
             color='green', label='Class 1 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--',
             color='blue', label='Class 2 vs Rest')
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('Multiclass ROC', dpi=300)
    st.pyplot()


def main():
    Nama = st.text_input('Nama Pemain')
    Klub = st.text_input('Klub Pemain')
    Kebangsaan = st.text_input('Kebangsaan Pemain')
    MinutesPlayed = st.text_input('Menit Bermain Pemain')
    Passing = st.text_input('Total Passing Pemain')
    Tackles = st.text_input('Total Tackles Pemain')
    Shots = st.text_input('Total Shots Pemain')
    Interception = st.text_input('Total Interception Pemain')

    data = pd.DataFrame({'Nama': ['Nama', 'Klub', 'Kebangsaan',
                        'MinutesPlayed', 'Passing', 'Tackles', 'Shots', 'Interception']})

    all_labelencoders = {}
    cols = ['Nama']
    for name in cols:
        labelencoder = LabelEncoder()
        all_labelencoders[name] = labelencoder

        labelencoder.fit(data[name])
        data[name] = labelencoder.transform(data[name])
        data_tr = data.transpose()

        prediksi = ''

        # membuat  button untuk prediksi
    if st.button('Prediksi Posisi Terbaik'):
        prediksi = prediksiketerangan(data_tr)
        prediction = clf.predict(data_tr)
        output = int(prediction[0])
        probas = clf.predict_proba(data_tr)
        output_probability = float(probas[:, output].round(3))
        result = {"confidence_score": output_probability}
        prediksi = prediksiketerangan(data_tr)
        st.success(prediksi)

        st.success(result)


if st.sidebar.checkbox('Prediksi Data Baru'):
    st.title('Inputkan Nilai Akademik Untuk Melakukan Klasifikasi')
    st.write(main())
