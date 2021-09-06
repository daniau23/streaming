from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.write("""
# Simple Iris Flower Prediction App
This app predicts the iris flower type!
""")

st.sidebar.header("User input Parameters")

def user_input_features():
    sepal_length = st.sidebar.slider("sepal_length", 4.3,7.9,5.4)
    sepal_width = st.sidebar.slider("sepal_width", 2.0,4.4,3.4)
    petal_length = st.sidebar.slider("petal_length", 1.0,6.9,1.3)
    petal_width = st.sidebar.slider("petal_width", 0.1,2.5,0.2)
    data = {"sepal_length": sepal_length,
            "sepal_width ": sepal_width ,
            "petal_length": petal_length,
            "petal_width ": petal_width 
            }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.subheader("User Input parameters")
st.write(df)

iris = load_iris()
X = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)

prediction = knn.predict(df)
prediction_proba = knn.predict_proba(df)

st.subheader("Class labels and their corresponding index number")
st.write(iris.target_names)


st.subheader("Prediction Probability")
st.write(prediction_proba)