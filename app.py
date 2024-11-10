import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('model_rf.pkl')

# Judul aplikasi
st.title("Air Quality Prediction Classifier")

# Input user
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Prediksi ketika tombol ditekan
if st.button("Predict"):
    # Buat prediksi
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    
    # Tampilkan hasil
    st.write(f"Prediction: {prediction[0]}")
