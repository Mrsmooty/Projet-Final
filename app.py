import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import load_model

def load_data(file, nrows):
    data= pd.read_csv(file)
    return data.head(nrows)

dataX = load_data("C:/Users/cleme/Desktop/Jedha/Projet de groupe/Prediction de sexe/CSV clean/X_train.csv",50)
dataY = load_data("C:/Users/cleme/Desktop/Jedha/Projet de groupe/Prediction de sexe/CSV clean/y_train.csv",947)
dataY = dataY.drop('id', axis=1)


# st.write("X Data", dataX)
# st.write("Y Data", dataY)

if 'label' in dataY.columns:
    
    dataY['gender_label'] = dataY['label'].map({0: 'Female', 1: 'Male'})
    g_count = dataY['gender_label'].value_counts()
    g_count_df = g_count.reset_index()
    g_count_df.columns = ['gender_label', 'count'] 

    
    fig_pie = px.pie(g_count_df, names='gender_label', values='count',
                     title='Distribution du Dataset', hole=0.4)

    # Display the chart using Streamlit
    #st.plotly_chart(fig_pie)

with h5py.File("C:/Users/cleme/Desktop/Jedha/Projet de groupe/Prediction de sexe/X_test_new.h5", 'r') as f:
    ls = list(f.keys())
    data = f.get('features')
    dataset = np.array(data)

model = load_model('C:/Users/cleme/Desktop/Jedha/Dashboard/CNN_model.h5')

def predict_sexe(data):
    predict = model.predict(data)
    return np.argmax(predict)

def main():
   
    option = st.selectbox('Choose the data view:', ('X Data', 'Y Data', 'Distribution Femme/Homme', 'Signal Plots', 'Prediction'))

    if option == 'X Data':
        st.write("X Data", dataX)

    elif option == 'Y Data':
        st.write("Y Data", dataY)

   
    elif option == 'Y Data Distribution':
        st.plotly_chart(fig_pie)

    
    elif option == 'Signal Plots':
        t = 2
        fr = 250
        x = [t / fr for t in range(len(dataset[0][0][0]))]
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 8))
        titles = ['sig0', 'sig1', 'sig2', 'sig3', 'sig4', 'sig5', 'sig6']

        for i in range(7):
            row, col = divmod(i, 4)
            sig = dataset[0][0][i]
            axes[row, col].plot(x, sig)
            axes[row, col].set_title(titles[i])

        plt.tight_layout()
        fig.suptitle('Les différents signaux', fontsize=15)

        if axes.shape[1] == 4:
            axes[1, 3].axis('off')

        st.pyplot(fig)

    elif option == 'Predction':
        predict = predict_sexe(data)
        st.write('Prediction', predict)
    

if __name__ == '__main__':
    main()

